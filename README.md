# Wave-VAE

git clone --recursive https://github.com/cwlgadd/CN-devel.git


To run supervised model:

python run_classifier.py <!--- -f <path_to>/<config_file>.yaml -->

# About this repo

## DataModules

1) SimulatedASCAT
   1) This dataset is constructed from an MDP with known kernel transition. 
2) ASCAT
   1) This dataset is publicly available genome count number variation data. 
   2) Our ML goal is to recover the genetic (assumed MDP kernel) mutations which led to this data

# Architectures

1) SequenceEncoder
2) Fully connected decoder
3) Wavelet RNN decoder

## The different models

1) **Classifier**. (Bidrectional) LSTM sequence classifier.
   1) Works on **both** data modules.
   2) Tests latent separability of cancer types when we have label information
   3) (Initial model to play with data)
2) **VAE**. Vanilla VAE with custom decoder 
   1) This is a testing model for use on **only** the simulatedASCAT data.
   3) This model has a simple 1-channel fully connected stacked linear encoder
   4) And the custom decoder outputs the coefficients of the kernel basis expansion. 
   5) This model then test our VAE construction under fully identifiable conditions
3) **WaveletVAE**
