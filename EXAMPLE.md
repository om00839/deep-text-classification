
## 사용 예시 (TextCNN)


```python
from utils import Instance, to_instances, Pipeline, EarlyStopping, evaluate, train, predict
from tokenizing import KhaiiiTokenizing
from embedding import W2vEmbedding
from label_encoding import SingleLabelEncoding
from data_loading import TextcnnDataLoading
from layers import W2vEmbeddingLayer, TextcnnRegularizedLayer
from metrics import accuracy

import os
import time
from datetime import datetime

import torch
from torch import nn
from torch import optim

# device = torch.device('cuda') # GPU 사용할 때
device = torch.device('cpu')
```

### 데이터 불러오기
#### Instance: 데이터 전처리 단계에서 사용되는 객체


```python
instances_train, n_label_train = to_instances('학습 데이터 경로')
instances_valid, n_label_valid = to_instances('테스트 데이터 경로')
```

### 전처리
#### Train set


```python
tokenizing = KhaiiiTokenizing() 
embedding = W2vEmbedding()
encoding = SingleLabelEncoding()
data_loading = TextcnnDataLoading(batch_size=32, torch.device('cpu'))
```


```python
instances_train = tokenizing(instances_train)
instances_train = encoding(instances_train)
instances_train = embedding(instances_train)
data_loader_train = data_loading(instances_train)
```

#### Validation set


```python
data_loading = TextcnnDataLoading(batch_size, device)
```


```python
instances_valid = tokenizing(instances_valid)
instances_valid = encoding(instances_valid)
data_loader_valid = data_loading(instances_valid)
```

### 모델 정의


```python
# Lookup Table Hyper-parameters
table_lookup_params = dict(index2word_dir='./model',
                           lookup_table_dir='./model', 
                           device=device)

# TextCNN Hyper-parameters
textcnn_params = dict(embedding_dim=300,
                      window_sizes=(3,4,5),
                      feature_map_size=100, 
                      dropout_rate=0.5,
                      class_size=48) 
              
# 모델 정의
class W2vTextcnnRegularized(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.table_lookup = W2vEmbeddingLayer(**table_lookup_params) # word embedding
        self.textcnn = TextcnnRegularizedLayer(**textcnn_params) # text cnn classifier
        
    def forward(self, batch):
        x = self.table_lookup(batch)
        logits = self.textcnn(x)
        return logits
    
model = W2vTextcnnRegularized().double().to(device)
```

### 모델 학습


```python
# optimizer, cost function 정의
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
criterion = nn.CrossEntropyLoss()
```

#### train


```python
PATH = '모델 저장/불러오기 경로'
```


```python
N_EPOCHS = 100

SAVE_PATH = PATH
print(f'SAVE PATH: {SAVE_PATH}')

patience = 5
verbose = True

early_stopping = EarlyStopping(SAVE_PATH, patience, verbose)

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, False, data_loader_train, optimizer, criterion)
    valid_loss = evaluate(model, data_loader_valid, criterion)
    
    early_stop = early_stopping(valid_loss, model)
    if early_stop:
        print(f'Epoch {epoch+1:02d}| Train Loss : {train_loss}, Val. Loss : {valid_loss}')
        break
        
    print(f'Epoch {epoch+1:02d}| Train Loss : {train_loss}, Val. Loss : {valid_loss}')
```

#### predict


```python
LOAD_PATH = PATH

train_true, train_pred, train_probs = predict(model,LOAD_PATH, data_loader_train, device)
valid_true, valid_pred, valid_probs = predict(model,LOAD_PATH, data_loader_valid, device)
```
