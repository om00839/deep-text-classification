"""
metrics.py
모델 성능평가시 사용되는 evaluation metric을 정의한 모듈
metric 수식 참고: A Literature Survey on Algorithms for Multi-label Learning
"""


import numpy as np

import torch

def exact_match_ratio(true, pred):
    """
    Args
    -----
        true (numpy.array)
        pred (numpy.array)
    """
    N = true.shape[0]
    equal = (true == pred)
    exact_match = equal.all(1).sum()
        
    return exact_match / N

def accuracy(true, pred):
    
    N = true.shape[0]
#     n_intersection = np.logical_and(true, pred).sum(1)
#     n_union = np.logical_or(true, pred).sum(1)
    accuracy = np.equal(true, pred).sum() / N
#     accuracy = (n_intersection / n_union).sum() / N
    
    return accuracy

def precision(true, pred):
    
    N = true.shape[0]
    n_intersection = np.logical_and(true, pred).sum(1)
    n_pred = pred.sum(1)
    
    precision = (n_intersection / n_pred).sum() / N
    
    return precision

def recall(true, pred):
    
    N = true.shape[0]
    n_intersection = np.logical_and(true, pred).sum(1)
    n_true = true.sum(1)
    
    recall = (n_intersection / n_true).sum() / N
    
    return recall

def f1_score(true, pred):
    
    N = true.shape[0]
    n_intersection = np.logical_and(true, pred).sum(1)
    n_pred = pred.sum(1)
    n_true = true.sum(1)
    
    f1_score = (2*n_intersection / (n_true + n_pred)).sum() / N
    
    return f1_score

def hamming_loss(true, pred):
    
    N = true.shape[0]
    n_xor = np.logical_xor(true, pred).sum(1)
    n_L = n_xor.shape[1]
    
    hamming_loss = (n_xor / n_L) / N
    
    return hamming_loss

def accuracy_topk(true, probs, k):
    
    N = true.shape[0]
    summation = 0
    for t, p in zip(true, probs):
        pred_topk = np.argpartition(p, -k)[-k:]
        n_correct = sum(map(lambda x: True if x in pred_topk else False, np.argwhere(t).flatten()))
        true_len = (t > 0).sum()
        if true_len < k:
            summation += (n_correct / true_len)
        else:
            summation += (n_correct / k)
            
    return summation / N
        
# 20190704 HY Version        
def top_k_acc(true, probs, k, num_label):
    N = len(true)
    n = num_label-1
    top_k = [probs[i].argsort()[-(k+n[i]):][::-1] for i in range(N)]
    top_k_acc = sum([np.isin(true[i], top_k[i]) for i in range(N)])/N
    return top_k_acc        
        
        
        
        