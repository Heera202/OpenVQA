from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from transformers import set_seed, GPT2Config, GPT2Tokenizer
from transformers import AutoTokenizer
from transformers.models.biogpt import BioGptTokenizer
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import pdb
import pickle

class medvqaDataset(Dataset):
  def __init__(self, path, split='train', like_test=False, prefix_length='gpt2'):
    super().__init__()
    data_path = path + split + '.pkl'
    with open(data_path, 'rb') as f:
      data = pickle.load(f)
    sys.stdout.flush()
