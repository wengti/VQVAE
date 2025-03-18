# Vector-Quantized Variational AutoEncoder (VQVAE)

## Credits & Acknoledgements
This project is a reimplementaion of [VQVAE-PyTorch] by [explainingai-code] (https://github.com/explainingai-code/VQVAE-Pytorch)
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.

## Features
- Build and train a **VQVAE** that is modifiable via config files that can:
  1. Take in either black and white or color images
  2. Accept various codebook sizes and latent dimensions
- Build and train a **LSTM** that can be used to generate a sequence of codebook indices for image generation.

## Description of Files:
- **extract_mnist.py** - Extracts MNIST data from CSV files.
- **load_data.py** - Create a custom dataset.
- **model.py** - Compatible with .yaml config files to create various VQVAE models.
- **engine.py** - Defines the train and test steps (for 1 epoch).
- **main.py** - Trains a VQVAE.
- **infer.py**
  1. Reconstruct images
  2. Extract and save encodings for LSTM training.
- **train_LSTM**
  1. Train a LSTM
  2. Apply the trained LSTM to generate indices for image generation.
