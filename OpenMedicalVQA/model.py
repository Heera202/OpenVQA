import torch
import torch.nn as  nn
import numpy as np

from typing import Tuple, Optional, Union
from torch.nn import fuctional as nnf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model
from prefix_mapper import MLP

class MedicalVQA(nn.Module):
  def __init__(
      self,
      prefix_length=8,
      clip_size = 512,
      embed_size = 768):

    super(MedicalVQA, self).__init__()

    self.mlp = MLP(
      sizes = [clip_size, 
        (prefix_length * embed_size) // 2, 
        prefix_length * embed_size,
        prefix_length * embed_size
        ]
    )

    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.gpt = GPT2LMHeadModel.from_pretrained("gpt2-xl", load_in_8bit=True, device_map = 'auto')
    self.gpt.eval()

    #projection for question tokens
    self.question_proj = nn.Linear(embed_size, embed_size)

    self.prefix_length = prefix_length
    self.embed_size = embed_size
    
  def forward(self, prefix, tokens, mask, q_len, labels=None):
    # 1. Project visual features
    prefix_projections = self.clip_project(prefix).view(
        -1, self.prefix_length, self.gpt_embedding_size
    )
    
    # 2. Embed tokens (supports BioGPT and others)
    embedding = (
        self.gpt.transformer.embed_tokens(tokens) 
        if self.gpttype == 'microsoft/biogpt' 
        else self.gpt.transformer.wte(tokens)
    )
    
    # 3. Vectorized visual token insertion
    batch_indices = torch.arange(tokens.size(0)).unsqueeze(1)
    token_indices = q_len.unsqueeze(1) + torch.arange(self.prefix_length, device=tokens.device)
    embedding[batch_indices, token_indices] = prefix_projections
    
    # 4. GPT forward (training/inference)
    return self.gpt(
        inputs_embeds=embedding,
        attention_mask=mask,
        labels = labels
    )

  def generate(self, prefix, tokens, mask, q_len):
    prefix_projections = self.clip_project(prefix.view(1, -1)).view(
        1, self.prefix_length, self.gpt_embedding_size
    )
    if self.gpttype == 'microsoft/biogpt':
        embeddings = self.gpt.transformer.embed_tokens(tokens)
    else:
        embeddings = self.gpt.transformer.wte(tokens)
       # 3. Insert visual prefix after question
    batch_indices = torch.arange(1, device=tokens.device).unsqueeze(1)
    token_indices = (q_len.view(-1) + torch.arange(self.prefix_length, device=tokens.device)).unsqueeze(0)
    embeddings[batch_indices, token_indices] = prefix_projections

    return embeddings
    











