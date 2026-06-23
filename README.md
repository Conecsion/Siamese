# Siamese Cryo-EM Particle Retrieval

Siamese contrastive encoder for matching low-SNR micrographs to clean projections.

## Installation

```bash
uv pip install -e .  # in existing .venv
```

## Quick Start

```bash
python scripts/generate_data.py emd_19110.map 8 --output-dir data/
python scripts/train.py --config configs/default.yaml
python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/
```