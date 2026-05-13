# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

# Transformer for German-to-English Translation

This repository contains a complete implementation of the Transformer architecture from scratch for German-to-English neural machine translation using the Multi30k dataset.

## Model Overview

The implementation includes:

- Multi-Head Self Attention
- Positional Encoding
- Encoder-Decoder Transformer Architecture
- Label Smoothing
- Noam Learning Rate Scheduler
- Greedy Decoding for Inference
- BLEU Score Evaluation

## Dataset

- Dataset: Multi30k
- Source Language: German
- Target Language: English
- Training Samples: 29,000
- Validation Samples: 1,014
- Test Samples: 1,000

## Training Configuration

- `d_model = 512`
- `N = 6`
- `num_heads = 8`
- `d_ff = 2048`
- `dropout = 0.1`
- `batch_size = 64`
- `num_epochs = 20`
- `warmup_steps = 4000`
- Optimizer: Adam (`betas=(0.9, 0.98)`, `eps=1e-9`)
- Learning Rate Scheduler: Noam Scheduler
- Loss Function: Label Smoothing (`ε = 0.1`)

## Performance

- Final Test BLEU Score: **32.13**

## Weights & Biases

Project Link: https://wandb.ai/ce23b124-indian-institute-of-technology-madras/da6401-assignment3

Report Link: *(to be added later)*

## Pretrained Model

The trained model weights (`best_model.pt`) are hosted on Google Drive and are automatically downloaded using `gdown` inside `Transformer.__init__()` if the checkpoint is not already present locally.

Google Drive File ID:

```text
1qqRwh6a84_qvGNvK50veDCjxVNkFz9Ly

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
