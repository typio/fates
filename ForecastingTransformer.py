#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

batch_size = 8
n_input = 24
n_output = 1
block_size = n_input + n_output
max_iters = 5000
eval_iters = 10
LR = 1e-3
split_n = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 1024
n_head = 2
n_layer = 1
dropout = 0.2

scaler = MinMaxScaler(feature_range=(-1, 1))

## Sine Wave Data
# def generate_sine_wave(length, freq=50, noise=0.05):
#     x = np.linspace(0, freq * np.pi, length)
#     x = np.sin(2 * np.pi * freq * x / length)
#     x += noise * np.random.randn(*x.shape)
    
#     x = scaler                            \
#         .fit_transform(x.reshape(-1, 1))  \
#         .reshape(-1)  
    
#     return x

# plt.title("Sine Wave Data")

# data = torch.tensor(generate_sine_wave(1000), dtype=torch.float32)

## Square Wave Data
# def generate_square_wave(length, freq=50, noise=0.05):
#     x = np.linspace(0, freq * np.pi, length)
#     x = np.sin(2 * np.pi * freq * x / length)
#     x = np.sign(x)
#     x += noise * np.random.randn(*x.shape)
    
#     x = scaler.fit_transform(x.reshape(-1, 1)).reshape(-1)
    
#     return x

# plt.title("Square Wave Data")

# data = torch.tensor(generate_square_wave(1000), dtype=torch.float32)

## Airline Passenger Data
airline_df = pd.read_csv('./data/airline-passengers.csv')
data = torch.tensor(airline_df['Passengers'], dtype=torch.float32)
data = torch.tensor(scaler.fit_transform(data.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
plt.title("Airline Passengers")

## Stock Data
# df = pd.read_csv("./data/stock_prices.csv")
# df["symbol"].unique()
# df = df[df["symbol"] == "SPY"]
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df = df.set_index('timestamp')
# df.shape

# def custom_resampler(array_like):
#     if len(array_like) == 0:
#         return None
#     return array_like.mean()

# # Resample the numeric columns
# resampled_numeric = df[['regularMarketPrice', 'regularMarketVolume']].resample('5S').apply(custom_resampler).dropna()

# # Resample the 'symbol' column
# resampled_symbol = df[['symbol']].resample('5S').apply(lambda x: x.iloc[0] if len(x) > 0 else None).dropna()

# # Combine the resampled data
# resampled_df = pd.concat([resampled_symbol, resampled_numeric], axis=1).reset_index()

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

# df = df.reset_index()

# df['regularMarketPrice'].plot(ax=axes[0], title="Original SPY Price", x=df.index)
# resampled_df['regularMarketPrice'].plot(ax=axes[1], title="Resampled SPY Price")

# plt.tight_layout()
# plt.show()

# data = torch.tensor(resampled_df['regularMarketPrice'], dtype=torch.float32)
# data = torch.tensor(scaler.fit_transform(data.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
# plt.title("Stock $SPY Price")

train_data, val_data = data[:int(len(data) * split_n)], data[int(len(data) * split_n):]

plt.plot(range(len(train_data)), train_data, label='Train Data')
plt.plot(range(len(train_data), len(train_data) + len(val_data)), val_data, label='Val Data')
plt.legend(loc="upper right")
plt.show()


# In[3]:


def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([torch.cat((data_set[i:i+n_input], torch.zeros(n_output))).unsqueeze(-1) for i in ix]).transpose(0, 1)
    y = torch.stack([data_set[i:i+block_size].unsqueeze(-1) for i in ix]).transpose(0, 1)
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')


# In[4]:


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesModel(nn.Module):
    def __init__(self, feature_size=n_embd, num_layers=n_layer, dropout=dropout):
        super(TimeSeriesModel, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoder(n_embd)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(n_embd, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


# In[5]:


@torch.no_grad()
def evaluate():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('val')
        logits = model(X)
        losses[k] = criterion(logits[-n_output:], Y[-n_output:]).item()
        loss = losses.mean()
    model.train()
    return loss


# In[6]:


model = TimeSeriesModel()
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

criterion = nn.HuberLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3, eta_min=1e-5)

loss_history = []
best_loss = float("inf")
best_model = None

for iter in range(max_iters):
    val_loss = evaluate()

    xb, yb = get_batch('train')

    logits = model(xb)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(logits[-n_output:], yb[-n_output:])
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model
        
    if iter % 100 == 0 or iter == max_iters - 1:
        print(f"step {iter}: train loss {loss:.4f}, val loss {val_loss:.4f}")    
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append({'train_loss': loss.item(), 'val_loss': val_loss})

## Hyperparameter grid search
# num_configs = len(n_embds) * len(n_heads) * len(n_layers)
# num_columns = 3 
# num_rows = int(np.ceil(num_configs / num_columns))
# fig = plt.figure(figsize=(16, num_rows * 6))


# n_embds = [64, 128]
# n_heads = [1, 2]
# n_layers = [1, 2]

# max_iters = 5000

# best_val_loss = float('inf')
# best_params = None

# subplot_idx = 1
# for n_e in n_embds:
#     for n_h in n_heads:
#         for n_l in n_layers:
#             n_embd = n_e 
#             n_head = n_h
#             n_layer = n_l
#             print(n_e, n_h, n_l)
#             model = TimeSeriesModel()
#             model = model.to(device)
            
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3, eta_min=1e-5)
#             optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
            
#             for iter in range(max_iters):
#                 xb, yb = get_batch('train')

#                 logits = model(xb)
#                 optimizer.zero_grad(set_to_none=True)
#                 loss = criterion(logits[-n_output:], yb[-n_output:])


#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
                
#                 # Evaluate the model on the validation set and store the best performing configuration
#                 val_loss = evaluate()
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     best_params = (n_e, n_h, n_l)
                    
#                 if iter % 100 == 0 or iter == max_iters - 1:
#                     print(f"step {iter}: train loss {loss:.4f}, val loss {val_loss:.4f}") 
                
#             context = train_data[:-n_input].unsqueeze(-1)
#             preds = generate(model, context, 100).cpu().squeeze()

#             # Create a new subplot and plot the predictions
#             ax = fig.add_subplot(num_rows, num_columns, subplot_idx)
#             ax.plot(range(len(train_data)), train_data, label='Train Data')
#             ax.plot(range(len(train_data), len(train_data) + len(val_data)), val_data, label='Val Data')
#             ax.plot(range(len(context)), context, label='Context')
#             ax.plot(range(len(context), len(context) + len(preds)), preds, label='Predictions', linestyle='--')
#             ax.set_title(f'Airline Passenger Dataset')
#             ax.set_xlabel('Months')
#             ax.set_ylabel('Passengers')
#             ax.legend(loc="lower right")

#             # Annotate the hyperparameters on the graph
#             hyperparams_str = f'n_embd={n_e}, n_head={n_h}, n_layer={n_l}'
#             ax.annotate(hyperparams_str, xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='black')

#             subplot_idx += 1
  
#             print("Best hyperparameters:", best_params)

# plt.tight_layout()
# plt.show()


# In[ ]:


torch.save(model.state_dict(), './forecast_transformer.pt')


# In[7]:


train_losses = [entry['train_loss'] for entry in loss_history]
val_losses = [entry['val_loss'] for entry in loss_history]

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()


# In[20]:


print(best_loss)
def generate(m, context, steps):
    context = context.to(device)
    m.eval()
    preds = torch.tensor([]).to(device)
    
    x = context.unsqueeze(-1)
    
    with torch.no_grad():
        for i in range(0, steps, 1):
            logits = m(x[-n_input:])
            x = torch.cat((x, logits[-1:]))
    return x[len(context):]
    
context = train_data[:-n_input].unsqueeze(-1)
preds = generate(best_model, context, 48).cpu().squeeze()


# In[21]:


context = context
plt.figure(figsize=(12, 6))
plt.title('Airline Passenger Data')

plt.plot(range(len(train_data)), train_data, label='Train Data', color='C0')
plt.plot(range(len(train_data), len(train_data) + len(val_data)), val_data, label='Val Data', color='C1')
plt.plot(range(len(context)), context, label='Context', color='C2')
plt.plot(range(len(context), len(context) + len(preds)), preds, label='Predictions', linestyle='--', color='C3')
plt.xlabel('Months')
plt.ylabel('Passengers')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




