# SB-ECC: Score-Based Error Correction Codes

A score-based neural decoder for linear block codes.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Project Structure

```
paper_submission/
├── Main.py          # Training script
├── SBECC.py         # SB_ECC model definition (CrossMPT backbone)
├── Codes.py         # Code utilities (generator/parity matrices)
├── solvers.py       # ODE solvers (Euler, DPM)
├── inference.py     # Inference utilities
├── train.sh         # Example training script
├── requirements.txt # Python dependencies
├── Codes_DB/        # Code database (generator/parity check matrices)
└── SB_ECC_Results/  # Training outputs and saved models
```

## Training

### Quick Start

Run the example training script:

```bash
bash train.sh
```

### Custom Training

Train a model with custom parameters:

```bash
python paper_submission/Main.py \
    --gpus=0 \
    --lr=0.0005 \
    --N_dec=6 \
    --d_model=128 \
    --code_type=POLAR \
    --code_n=64 \
    --code_k=32 \
    --solver=euler \
    --test_batch_size=2048 \
    --batch_size=256 \
    --epochs=1500 \
    --sigma_max=0.8 \
    --sigma_min=0.1
```

### Training Arguments

#### General Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 1500 | Number of training epochs |
| `--lr` | 5e-4 | Learning rate |
| `--batch_size` | 128 | Training batch size |
| `--test_batch_size` | 2048 | Testing batch size |
| `--gpus` | 0 | GPU device ID(s) |
| `--workers` | 4 | Number of data loader workers |
| `--seed` | 42 | Random seed |

#### Code Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--code_type` | POLAR | Code type: `BCH`, `POLAR`, `LDPC`, `CCSDS`, `MACKAY` |
| `--code_n` | 64 | Code block length (n) |
| `--code_k` | 32 | Code dimension / message length (k) |

#### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--N_dec` | 6 | Number of CrossMPT decoder layers |
| `--d_model` | 128 | Transformer model dimension |
| `--h` | 8 | Number of attention heads |

#### SDE / Diffusion Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--solver` | euler | ODE solver: `euler` or `dpm` |
| `--sigma_min` | 0.01 | Minimum noise standard deviation |
| `--sigma_max` | 0.8 | Maximum noise standard deviation |

## Output

Training outputs are saved to `SB_ECC_Results/<code_type>__Code_n_<n>_k_<k>__<timestamp>/`:
- `logging.txt` - Training and evaluation logs
- `best_model` - Best model checkpoint (saved when training loss improves)

## Supported Codes

The decoder supports the following linear block codes:
- **POLAR** - Polar codes
- **BCH** - BCH codes
- **LDPC** - Low-Density Parity-Check codes
- **CCSDS** - CCSDS standard codes
- **MACKAY** - MacKay LDPC codes

## Acknowledgments

This project is based on:
- **[DDECC](https://github.com/yoniLc/DDECC)** - We forked the original DDECC repository as the foundation for this work
- **[CrossMPT](https://github.com/iil-postech/crossmpt)** - We adopted the CrossMPT architecture as our model backbone.
