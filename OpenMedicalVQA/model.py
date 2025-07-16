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

    super().__init__()

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
    
  
  def forward(self, prefix, question_ids, answer_ids=None):
    visual_prefix = self.mlp(prefix).view(-1, self.prefix_length, self.embed_size)

    #process question tokens
    question_embeds = self.gpt.transformer.wte(question_ids)
    question_embeds = self.question_proj(question_embeds)

    #combine outputs
    inputs_embeds = torch.cat([
      self.gpt.transformer.wte(torch.tensor([[self.tokenizer.bos_token_id]], device = prefix.device)),
      question_embeds,
      visual_prefix,
      self.gpt.transformer.wte(torch.tensor([[self.tokenizer.eos_token_id]], device = prefix.device))
    ], dim=1)

    outputs = self.gpt(
      inputs_embeds = inputs_embeds,
      labels = answer_ids
    )

    return outputs


  def generate(
    self, 
    prefix, 
    question_ids, 
    max_length=50,
    num_beams = 5,
    temparature=1.0,
    top_k = None,
    top_p = None, 
    answer_ids=None,
    method ="beam", 
    repetition_penalty=1.0):

    visual_prefix = self.mlp(prefix).view(-1, self.prefix_length, self.embed_size)

    question_embeds =  self.gpt.transformer.wte(question_ids)
    question_embeds = self.question_proj(question_embeds)

    #combine inputs with BOS/EOS tokens
    batch_size = question_ids.shape[0]
    bos_token = torch.tensor([[self.tokenizer.bos_token_id]], device=prefix.device)
    eos_token = torch.tensor([[self.tokenizer.eos_token_id]], device=prefix.device)

    inputs_embeds = torch.cat([
      self.gpt.transformer.wte(bos_token).expand(batch_size, -1, -1),
      question_embeds,
      visual_prefix,
      self.gpt.transformer.wte(eos_token).expanf(batch_size, -1, -1) 
    ], dim=1)


    # --Method-Specific Generation ---
    if method in ['beam', 'greedy']:
      generated_ids = self.gpt.generate(
        inputs_embeds = inputs_embeds,
        max_length = max_length,
        bum_beams = num_beams if method == "beam" else 1,
        temparature = temparature,
        top_k = top_k,
        top_p = top_p,
        pad_token_id = self.tokenizer.eos_token_id
      )

    elif method == "custom_beam":
      generated_ids = self.generate_beam(
        inputs_embeds = inputs_embeds,
        beam_size = num_beams,
        max_length = max_length

      )

    return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)












