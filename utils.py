"""
utils.py
다른 모듈들에 공통적으로 클래스, 메소드를 정의한 모듈
"""

import os
import time
import torch
import random

import numpy as np
import pandas as pd



from typing import NamedTuple
from tqdm import tqdm_notebook

from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import data_parallel
from torch.autograd import Variable

# 전처리 과정에서 각 샘플을 관리하기 위해 만든 NamedTuple
# csv 파일에서 상담ID, 상담내용(텍스트), 최종분류코드(라벨)만 사용
class Instance(NamedTuple):
    """
    Instance
    Attribute:
        id(str): 상담ID
        text(str): 상담내용
        label(str): 최종분류코드
    """
    id: str
    text: object
    label: object
        

# csv파일에서 list of instances로 loading
    """
    args:

    data_file: 파일경로
    id_col: 상담ID
    text_col: 상담내용
    label_col: 최종분류코드
    sep: 구분자
"""
def to_instances(data_file, id_col='상담ID', text_col='상담내용', label_col='최종분류코드', sep=',', num_code=True):
    data = pd.read_csv(data_file, sep=sep) # data 읽어오기
    
    if num_code:
        num_code = data['분류코드수'] # 190704 수정
    else:
        num_code = np.repeat(1,len(data)).tolist()
        
    instances = [Instance(id_, text, label) 
                 for id_, text, label 
                 in zip(data[id_col], data[text_col], data[label_col])]
    return instances, num_code


# 전처리 과정의 각 클래스를 정의하기 위한 interface
# 전처리 과정의 각 클래스는 Pipeline을 상속받아 __init__ function과 __call__ function을 override 해야함.
class Pipeline:
    """Pipeline Class : callable"""
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError()


# pytorch 기반의 model을 학습시키기 위한 함수
def train(model, data_parallel, data_loader, optimizer, criterion):
    """
    train()
    -------
    Args:
        model(nn.Module): pytorch nn.Module을 상속받아 만든 모델 객체
        data_parallel(bool): Multi-GPU를 사용할지 말지 여부
        data_loader(torch.utils.data.DataLoader): batch iterator
        optimizer(torch.optim): 학습 알고리즘 (ex. Adam, SGD, etc...)
        criterion(nn.Loss): loss function
    """
    model.train()
    if data_parallel: # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model)
    epoch_loss = 0
    iter_bar = tqdm_notebook(data_loader, desc='Iter (loss=X.XXX)')
    
    for i, batch in enumerate(iter_bar):
        
        optimizer.zero_grad()
        
        id_, X, y = batch
                
        logits = model(X)
        
        loss = criterion(logits, y)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        iter_bar.set_description(f'Iter (loss={loss.item():.3f})')
        
    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    """
    evaluate()
    -------
    Args:
        model(nn.Module): pytorch nn.Module을 상속받아 만든 모델 객체
        data_loader(torch.utils.data.DataLoader): batch iterator
        criterion(nn.Loss): loss function
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            id_, X, y = batch
            
            logits = model(X)
            
            loss = criterion(logits, y)
            epoch_loss += loss.item()
            
    return epoch_loss / len(data_loader)


def predict(model, load_path, data_loader, device):
    """
    predict()
    -------
    Args:
        model(nn.Module): pytorch nn.Module을 상속받아 만든 모델 객체
        load_path(str): 이미 학습된 model (state_dict)
        data_loader(torch.utils.data.DataLoader): batch iterator
        device(torch.device): 모델 연산시 cpu를 사용할지, gpu를 사용할지
    """
    # load model
    model.load_state_dict(torch.load(load_path))
    print('Loading the model from', load_path)
    
    model.eval()
    epoch_loss = 0
    
    true = list()
    pred = list()
    log_probs = list()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            id_, X, y = batch
            
            true.append(y)
            logits = model(X)
            # logits shpae: (batch_size, class_size)
            
#             pred.append(torch.tensor((logits > 0), device=y.device, dtype=y.dtype)) # CrossEntropy로 인한 수정
            _, pred_y = logits.max(1)
            pred.append(pred_y)
#             log_probs.append(F.logsigmoid(logits))
            log_probs.append(F.softmax(logits))
                        
            
    true = torch.cat(true).cpu().numpy()
    pred = torch.cat(pred).cpu().numpy()
    log_probs = torch.cat(log_probs).cpu().numpy()
            
    return true, pred, log_probs


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False):
        """
        Args:
            patience (int): How long to wait agter last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improved.
                            Default: False
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        
        score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            
        elif score > self.best_score:
            self.counter += 1 
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            torch.save(model.state_dict(), self.save_path)
            print("Saving the model to", self.save_path)
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
# 아래는 pytorch-bert에서 가져온 소스코드    
def load_pretrained_model(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts
                
def norm(instances_train, instances_valid):
    """Instances Normalization for MultiLayerPerceptron Training"""
    
    """
    Args:
        instances_train (list): instances_train(id, text, label)
                        Default: None
        instances_valid (list): instances_valid(id, text, label)
                        Default: None
    """
    doc_vec = list(map(lambda x: x[1], instances_train))
    tr_mean = torch.stack(doc_vec).mean(0)
    tr_std = torch.stack(doc_vec).std(0)
    
    instances_tr = []
    for id_, text, label in instances_train:
        document = (text - tr_mean)/tr_std
        instances_tr.append(Instance(id_, document, label))
        
    instances_va = []
    for id_, text, label in instances_valid:
        document = (text - tr_mean)/tr_std
        instances_va.append(Instance(id_, document, label))
        
    return instances_tr, instances_va

def norm_doc(instances_train, instances_valid):
    """Instances Normalization for MultiLayerPerceptron Training"""
    
    """
    Args:
        instances_train (list): instances_train(id, text, label)
                        Default: None
        instances_valid (list): instances_valid(id, text, label)
                        Default: None
    """
    tr = list(map(lambda x: x[1][0], instances_train))
    tr_mean = np.mean(tr, axis=0)
    tr_std = np.std(tr,axis=0)
    
    instances_tr = []
    for id_, text, label in instances_train:
        document = (text - tr_mean)/tr_std
        instances_tr.append(Instance(id_, document, label))
        
    instances_va = []
    for id_, text, label in instances_valid:
        document = (text - tr_mean)/tr_std
        instances_va.append(Instance(id_, document, label))
        
    return instances_tr, instances_va


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

