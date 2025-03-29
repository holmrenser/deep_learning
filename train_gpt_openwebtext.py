# %% [markdown]
# # GPT from scratch (but very very small)
# In this notebook we will build a GPT-style transformer from scratch. Heavily based on [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT/tree/master) by Andrej Karpathy.
# 
# Emphasis on readable code, minimal and simple implementations, and (relatively) fast training.

# %%
import math
import pickle
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler, random_split, IterableDataset

from datasets import load_dataset
from train_gpt import GPT, CharacterTokenizer


DEVICE = torch.device('cuda:1') # 'mps' for ARM macbooks, 'cuda' for colab, 'cpu' otherwise
DATASET = 'openwebtext'


class OWTDataset(IterableDataset):
  def __init__(self, tokenizer, context_size:int=64):
    super(OWTDataset).__init__()
    self.dataset = load_dataset('Skylion007/openwebtext', streaming=True, split='train')
    self.tokenizer = tokenizer
    self.context_size = context_size


  def __iter__(self):
    remainder = torch.empty(0)
    for d in self.dataset:
      tokens = self.tokenizer.encode(d['text'].replace('\n',' ') + '\n')
      tokens = torch.tensor(tokens, dtype=torch.long)
      if remainder.numel():
        chunk = torch.cat([remainder, tokens[:self.context_size - len(remainder) + 1]])
        yield chunk[:-1], chunk[1:]
      for pos in range(len(remainder), len(tokens), self.context_size):
        chunk = tokens[pos: pos + self.context_size + 1]
        if len(chunk) < self.context_size + 1:
          remainder = chunk
        else:
          yield chunk[:-1], chunk[1:]

if __name__ == '__main__':
    train_steps = 200_000
    batch_size = 64
    context_size = 512
    n_layers = 12
    n_heads = 12
    embedding_dim = 384
    learning_rate = 1e-3
    train_fraction = 0.9
    dropout = 0.1

    tokenizer = CharacterTokenizer()

    train_dataset = OWTDataset(tokenizer=tokenizer, context_size=context_size)
    print(train_dataset)

    model = GPT(context_size=context_size, tokenizer=tokenizer, n_layers=n_layers, embedding_dim=embedding_dim, n_heads=n_heads, dropout=dropout)
    # model = torch.compile(model)
    print(model)
    try: 
        n_epochs = (train_steps * batch_size) / len(train_dataset)
        print(f'Training for {train_steps=}, {n_epochs=:.3}')
    except:
        print(f'Training for {train_steps=}')

    model.to(DEVICE)
    model.train()

    # train_dataset, test_dataset = random_split(dataset, [train_fraction, 1 - train_fraction])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=train_steps)
    model.train()

    train_losses = []
    train_accuracies = []

    for i, (train_x, train_y) in enumerate(tqdm(train_dataloader, total=train_steps)):
        if i == train_steps - 1:
            break
        # forward the model
        try:
            _,train_loss,train_accuracy = model(train_x.to(DEVICE), train_y.to(DEVICE))
        except Exception as err:
            print(train_x.shape, train_y.shape)
            raise err
            

        # save losses on train and test every 20 iterations
        if i % 20 == 0:
            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy.item())

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        train_loss.backward()

        # Prevent gradients from becoming too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.save(f'{DATASET}.model.pkl')

    print(f'Final loss = {train_losses[-1]:.3}')


    print(model.generate())


