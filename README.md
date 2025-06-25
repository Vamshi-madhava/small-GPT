# miniGPT: A Lightweight Transformer-Based Language Model

A compact GPT-style language model built from scratch using PyTorch.  
Supports both pretraining on BookCorpus-like datasets and fine-tuning for question answering (QA) tasks.

---

Overview

This project demonstrates:
- Building a decoder-only transformer (GPT) model from scratch.
- Pretraining on a BookCorpus-style `.bin` dataset.
- Fine-tuning on a custom QA-style dataset.
- Generation using learned language patterns.


 Model Architecture

  Parameter        Value        

  Layers           12           
  Attention Heads  12            
  Embedding Size   768         
  Context Length   256          
  Vocabulary       50,257 (GPT2Tokenizer)  

> The model architecture is inspired by GPT-1/GPT-2, but scaled down for Colab-level training.
> PyTorch module is used in this project. 
> I haven't uploaded the dataset I used for training because the dataset size is too big for a GitHUb repo.

