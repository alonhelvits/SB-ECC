#!/bin/bash
# SB-ECC Training Script

python Main.py --gpus=0 \
    --lr=0.0005 \
    --N_dec=6 \
    --d_model=128 \
    --code_type=BCH \
    --code_n=63 \
    --code_k=45 \
    --solver=euler \
    --test_batch_size=2048 \
    --batch_size=256 \
    --epochs=1500 \
    --sigma_max=0.8 \
    --sigma_min=0.1
