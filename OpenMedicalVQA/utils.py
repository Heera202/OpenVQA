import torch
from tqdm import tqdm
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

def treebank_tokenize(s):
  return TreebankWordTokenizer().tokenize(s)

def generate_beam(
  model,
  tokenizer,
  beam_size: int = 5,
  generated = None,
  entry_length = 65,
  temparature = 1.0,
  stop_token: str = "<|endoftext|>",
):
  model.eval()
  stop_token_index = tokenizer.encode(stop_token)[0]                        #EOS token ID
  tokens = None                                                             #will store beam sequences
  scores = None                                                             #Cumulative log-probabilities
  device = next(model.parameters()).device
  seq_lengths = torch.ones(beam_size, device = device)
  is_stopped = torch.zeros(beam_size, device = device, dtype=torch.bool)

  with torch.no_grad():
    for i in range(entry_length):
      outputs = model.gpt(inputs_embeds = generated)                        #Logits shape: [1, seq_len, vocab_size]
      logits = outputs.logits 
      logits = logits[:, -1, :] / (temparature if temparature > 0 else 1.0) #Last tokem logits: [1, vocab_size] and temparature scaling
      logits = logits.softmax(-1).log()                                     #covert to log-probs

      if scores is None:

        scores, next_tokens = logits.topk(beam_size, -1)                        #Top 3 tokens
        generated = generated.expand(beam_size, *generated.shape[1:])           #Duplicated for beams
        next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)       #shape: [3, 1](3 beams, 1 token each)
        
        if tokens is None:
          tokens = next_tokens
        else:
          tokens = tokens.expand(beam_size, *tokens.shape[1:])
          tokens = torch.cat((tokens, next_tokens), dim=1)

      else:
        logits[is_stopped] = -float(np.inf)
        logits[is_stopped, 0] = 0
        scores_sum = scores[:, None] + logits
        seq_lengths[~is_stopped] += 1
        scores_sum_average = scores_sum / seq_lengths[:, None]
        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
          beam_size, -1
        )
        next_tokens_source = next_tokens // scores_sum.shape[1]
        seq_lengths = seq_lengths[next_tokens_source]
        next_tokens = next_tokens % scores_sum.shape[1]
        tokens = torch.cat((tokens[next_tokens_source], next_tokens.unsqueeze(1)), dim=1)
        generated = generated[next_tokens_source]
        scores = scores_sum_average * seq_lengths
        is_stopped = is_stopped[next_tokens_source]
      
        next_token_embed = model.gpt.get_input_embeddings()(tokens[:, -1])
        next_token_embed = next_token_embed.squeeze().view(generated.shape[0], 1, -1)

      generated = torch.cat((generated, next_token_embed), dim = 1)
      is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
      if is_stopped.all():
        break
    
  scores = scores / seq_lengths
  output_list = tokens.cpu().numpy()
  output_texts = [
    tokenizer.decode(output[: int(length)])
    for output, length in zip(output_list, seq_lengths)
  ]

  order = scores.argsort(descending = True)
  output_texts = [output_texts[i] for i in order]
  return output_texts
  



