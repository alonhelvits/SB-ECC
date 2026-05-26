"""
SB-ECC Inference Script
Load a trained model and evaluate on test data.
"""
from __future__ import print_function
import argparse
import os
from torch.utils.data import DataLoader
import logging
from Codes import *
import time
from Main import set_seed, FEC_Dataset
import torch
import numpy as np


def test(model, device, test_loader_list, EbNo_range_test, min_FER=500, max_cum_count=1e9, 
         min_cum_count=1e5, solver_type='euler', num_steps=10):
    model.eval()
    test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], []
    t = time.time()
    
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_fer = cum_count = np.float64(0.0)
            
            # Initialize with dummy forward pass
            _, x_pred_list, _, _ = model.p_sample_loop(
                next(iter(test_loader))[3].to(device), 
                solver_type=solver_type, num_steps=num_steps)
            test_ber_ddpm = [0] * len(x_pred_list)
            test_fer_ddpm = [0] * len(x_pred_list)
            
            idx_conv_all = []
            next_print_count = 1e8
            
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                x_pred, x_pred_list, idx_conv, synd_all = model.p_sample_loop(
                    y, solver_type=solver_type, num_steps=num_steps)
                x_pred = sign_to_bin(torch.sign(x_pred))

                idx_conv_all.append(idx_conv)
                for kk, x_pred_tmp in enumerate(x_pred_list):
                    x_pred_tmp = sign_to_bin(torch.sign(x_pred_tmp))
                    x = x.to(x_pred_tmp.device)
                    test_ber_ddpm[kk] += BER(x_pred_tmp, x) * x.shape[0]
                    test_fer_ddpm[kk] += FER(x_pred_tmp, x) * x.shape[0]
                    
                test_ber += BER(x_pred, x) * x.shape[0]
                test_fer += FER(x_pred, x) * x.shape[0]
                cum_count += x.shape[0]

                # Periodic logging
                if cum_count >= next_print_count:
                    current_ber = test_ber / cum_count
                    current_fer = test_fer / cum_count
                    current_ber_ddpm = [b / cum_count for b in test_ber_ddpm]
                    current_fer_ddpm = [f / cum_count for f in test_fer_ddpm]

                    logging.info(f'Intermediate (Samples: {cum_count:.2e}) Test EbN0={EbNo_range_test[ii]}')
                    logging.info(f'BER={current_ber}')
                    logging.info(f'FER={current_fer}')
                    logging.info(f'BER_DDPM={current_ber_ddpm}')
                    logging.info(f'-ln(BER)_DDPM={[-np.log(elem) if elem > 0 else np.inf for elem in current_ber_ddpm]}')
                    logging.info(f'FER_DDPM={current_fer_ddpm}')
                    next_print_count += 1e8

                # Check stopping criteria
                if (min_FER > 0 and test_fer > min_FER and cum_count > min_cum_count) or cum_count >= max_cum_count:
                    if cum_count >= 1e9:
                        logging.info(f'Cum count reached EbN0:{EbNo_range_test[ii]}')
                    else:    
                        logging.info(f'FER count threshold reached EbN0:{EbNo_range_test[ii]}')
                    break
                    
            idx_conv_all = torch.stack(idx_conv_all).float()
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            
            for kk in range(len(test_ber_ddpm)):
                test_ber_ddpm[kk] /= cum_count
                test_fer_ddpm[kk] /= cum_count
                
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER_DDPM={test_ber_ddpm}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, -ln(BER)_DDPM={[-np.log(elem) for elem in test_ber_ddpm]}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, FER_DDPM={test_fer_ddpm}')
            logging.info(f'#It. to zero syndrome: Mean={idx_conv_all.mean():.2f}, Std={idx_conv_all.std():.2f}, Min={idx_conv_all.min()}, Max={idx_conv_all.max()}')
            
        # Summary
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in zip(test_loss_fer_list, EbNo_range_test)]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in zip(test_loss_ber_list, EbNo_range_test)]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.3e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in zip(test_loss_ber_list, EbNo_range_test)]))
             
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_fer_list


def inference_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    logging.info('Loading Model For Inference')
    model = torch.load(args.model_path, weights_only=False)
    model.to(device)

    # Setup code
    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code

    # Parse EbN0 range
    ebno_values = [int(x.strip()) for x in args.ebno_range.split(',')]
    if len(ebno_values) == 2:
        EbNo_range_test = range(ebno_values[0], ebno_values[1])
    else:
        EbNo_range_test = ebno_values
        
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    test_dataloader_list = [
        DataLoader(FEC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                   batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers)
        for ii in range(len(std_test))]
    
    logging.info(f'Code: {code.code_type} n={code.n}, k={code.k}')
    logging.info(f'Solver: {args.solver}, Steps: {args.num_steps}')
    
    # Run evaluation
    max_cum_count = args.max_samples if args.max_samples is not None else 1e8
    min_FER = 0 if args.max_samples is not None else 500
    
    test(model, device, test_dataloader_list, EbNo_range_test, 
         max_cum_count=max_cum_count, min_FER=min_FER, 
         solver_type=args.solver, num_steps=args.num_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SB-ECC Inference')
    
    # Code args
    parser.add_argument('--code_type', type=str, default='POLAR', help='Code type')
    parser.add_argument('--code_n', type=int, default=64, help='Code length')
    parser.add_argument('--code_k', type=int, default=48, help='Code dimension')
    
    # Model args
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    
    # Inference args
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'dpm'],
                        help='ODE solver type')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of denoising steps')
    parser.add_argument('--ebno_range', type=str, default='4,7',
                        help='EbN0 range: "start,end" or comma-separated list')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to test (overrides min_FER)')
    
    # Runtime args
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids')
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)

    handlers = [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    logging.info(args)

    inference_main(args)
