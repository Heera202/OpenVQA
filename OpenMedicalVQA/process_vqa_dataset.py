import torch
import skimage.io as io
import skimage.transform as transform
import torchvision
import clip
import pandas as pd
from PIL import Image
import pickle
import json 
import os
from tqdm import tqdm 
import argsparse
import string
import random
import numpy as np 
from transformers import set_seed, GPT2Config, GPT2Tokenizer


def isEglish(s):
  return s.isascii()

def punc(s):
  for c in string.punctuation:
    s = s.replace(c, "")
  return s.lower()

def update_classes(pkl_train, pkl_val, pkl_test):
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  with open(pkl_train, 'rb') as f:
    data_train = pickle.load(f)
  with open(pkl_val, 'rb') as f:
    data_val = pickle.load(f)
  with open(pkl_test, 'rb') as f:
    data_test = pickle.load(f)

  cur_id = 0
  class_names_list = []
  class_ids_list = [[], [], []]
  for i, data in enumerate([data_train, data_val, data_test]):
    for answer in data['answers']:
      if answer not in class_names_list:
        class_names_list.append(answer)
        class_ids_list[i].append(cur_id)
        cur_id += 1
      else:
        class_ids_list[i].append(class_names_list.index(answer))
  q_lens = []
  a_lens = []
  for question in data_train['questions']:
    q_lens.append(len(tokenizer.encode(question))) 
  for answer in data_train['answers']:
    a_lens.append(len(tokenizer.encode(str(answer))))

  
  for data, ids in zip([data_train, data_val, data_test], class_ids_list):
    data.update({'class_ids': ids, 'class_names':class_names_list})

  max_len = lambda lens: int(np.mean(lens) + 2*np.std(lens))
  max_seqs_len = (max_len(q_lens), max_len(a_lens))
  for data in [data_train, data_val, data_test]:
    data['max_seq_len'] = max_seqs_len

  with open(pkl_train, 'wb') as f:
    pickle.dump(data_train, f)
  with open(pkl_val, 'wb') as f:
    pickle.dump(data_val, f)
  with open(pkl_test, 'wb') as f:
    pickle.dump(data_test, f)


def preprocess_slake(split, out_path):
  device = torch.device('cuda:0')
  clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
  with open('..vqa_datasets/slake/Slake1.0/{}.json'.format(split)) as f:
    data = json.load(f)
  print("%d captions laoded from json " % len(data))

  all_img_prefixes = []
  img_ids = []
  img_paths = []
  all_questions = []
  all_answers = []
  img_dict = []

  for i in tqdm(range(len(data))):
    d = data[i]
    if isEglish(d['answer']) and isEglish(d['question']):
      img_id = d['img_id']
      filename = "../vqa_datasets/slake/Slake1.0/imgs/"+d['img_name']
      with torch.no_grad():
        prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()
      if img_id not in img_dict.keys():
        img_dict[img_id] = [[d['question']], [d['answer']], prefix_i, filename]
      else:
        img_dict[img_id][0].append(d['question'])
        img_dict[img_id][1].append(d['answer'])

  for img_id, imgs in enumerate(img_dict.keys()):
    all_img_prefixes.append(img_dict[imgs][2])
    for q in range(len(img_dict[imgs][0])):
      all_questions.append(img_dict[imgs][0][q])
      all_answers.append(img_dict[imgs][1][q])
      img_ids.append(img_id)
      img_paths.append(img_dict[imgs][3])
  
  all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions, 'answers' : all_answers, 'img_path':img_paths}
  with open(out_path, 'wb') as f:
    pickle.dump(all_data, f)
  print('Done')
  print("%0d embeddings saved " % len(all_questions))

if __name__ == '__main__':
  for split in ['train', 'test']:
    out_path = '../vqa_datasets/slake/{}.pkl'.format(split)
    preprocess_slake(split, out_path)
    update_classes()
        


