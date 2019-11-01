"""
layers.py
torch 기반의 low-level layer들을 흔히 사용하는 classifier, embedding등 high-level layer로 정의한 모듈
"""


from utils import split_last, merge_last

import os
import json
import pickle
import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# multi layer perceptron layer
class MlpLayer(nn.Module):
    """
    MlpLayer Parameters
    -------------------
    embedding_dim : int, default: 300
        The dimension of input vector
    hidden_layer_dim : int, default: 300
        The dimension of hidden layer
    hidden_layer_size : int, default: 5
        The ith element represents the number of neurons in the ith
        hidden layer.
    class_size : int, default: 48
        The number of classes
    """
    def __init__(self, hidden_layer_size=5, embedding_dim=300, hidden_layer_dim=300, class_size=48):
        super().__init__()
        
        if hidden_layer_size < 1:
            raise ValueError()
            
        self.input_layer = nn.Linear(embedding_dim, hidden_layer_dim).double()
        
        hidden_layer_list = list()
        for i in range(hidden_layer_size-1):
            hidden_layer_list += [nn.ReLU(), nn.Linear(hidden_layer_dim, hidden_layer_dim)]
        self.hidden_layers = nn.Sequential(*hidden_layer_list).double()
        
        self.output_layer = nn.Linear(hidden_layer_dim, class_size).double()
#         self.output_layer = nn.Softmax(hidden_layer_dim, class_size).double() # 6.21 수정
        
    def forward(self, x):
        
        h = self.input_layer(x)
        h = self.hidden_layers(h)
        o = self.output_layer(h)
        
        return o
    

# multi layer perceptron layer + dropout layer    
class MlpDropoutLayer(nn.Module):
    """
    MlpLayer Parameters
    -------------------
    embedding_dim : int, default: 300
        The dimension of input vector
    hidden_layer_dim : int, default: 300
        The dimension of hidden layer
    hidden_layer_size : int, default: 5
        The ith element represents the number of neurons in the ith
        hidden layer.
    dropout_rate : float, default: 0.5
    class_size : int, default: 48
        The number of classes
    """
    def __init__(self, hidden_layer_size=2, embedding_dim=300, hidden_layer_dim=512, dropout_rate=0.5, class_size=48):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        if hidden_layer_size < 1:
            raise ValueError()
            
        self.input_layer = nn.Linear(embedding_dim, hidden_layer_dim).double()
        
        hidden_layer_list = list()
        for i in range(hidden_layer_size-1):
            hidden_layer_list += [nn.ReLU(), nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.Dropout(dropout_rate)]
        self.hidden_layers = nn.Sequential(*hidden_layer_list).double()
        
        self.output_layer = nn.Linear(hidden_layer_dim, class_size).double()
        
    def forward(self, x):
        
        x = F.dropout(x, self.dropout_rate)
        h = self.input_layer(x)
        h = self.hidden_layers(h)
        h = F.dropout(h, self.dropout_rate)
        o = self.output_layer(h)
        
        return o
    

# word embedding: token index -> word vector table lookup   
class W2vEmbeddingLayer(nn.Embedding):
    def __init__(self, index2word_dir, lookup_table_dir, d2v_model, device):
        
        self.device = device
        
        # load lookup_table
        with open(os.path.join(lookup_table_dir, d2v_model + 'w2v_lookup_table.pickle'), 'rb') as f:
            lookup_table = pickle.load(f)
            
        # load index2word
        with open(os.path.join(index2word_dir, d2v_model + 'w2v_index2word.json'), 'r') as f:
            index2word = json.load(f)
        
        # add special tokens to index2word
        special_tokens = ['<UNK>', '<PAD>'] # <UNK>: Unkown Token, <PAD>: Padding Token
        index2word = special_tokens + index2word
        self.vocab = set(index2word)
        
        # make word2index
        self.word2index = {word:index for index, word in enumerate(index2word)}
        
        # add special token's vectors to lookup table
        unk_vector = np.random.randn(1,lookup_table.shape[1])*0.01
        pad_vector = np.zeros(shape=(1,lookup_table.shape[1]))
        self.lookup_table = np.concatenate((unk_vector, pad_vector, lookup_table), axis=0)
        
        super().__init__(self.lookup_table.shape[0], self.lookup_table.shape[1])
        self.weight.data.copy_(torch.FloatTensor(self.lookup_table).to(device))
        
    def _get_max_doc_len(self, documents):
        max_doc_len = max(map(lambda x: len(x.split(' ')), documents))
        return max_doc_len
        
    def forward(self, documents):
        max_doc_len = self._get_max_doc_len(documents)
        indices = list()
        for document in documents:
            document_splitted = document.split(' ')
            words = document_splitted + (['<PAD>']*(max_doc_len - len(document_splitted)))
            temp = list()
            for word in words:
                index = self.word2index['<UNK>']
                if word in self.vocab:
                    index = self.word2index[word]
                temp.append(index)
            indices.append(temp)
        indices = torch.LongTensor(indices).to(self.device)
        output = super().forward(indices).unsqueeze(1)
        # output shape: (batch_size, 1, max_doc_len, embedding_dim)
        return  output
    

# TextCNN Layer    
class TextcnnLayer(nn.Module):
    def __init__(self, embedding_dim, window_sizes, feature_map_size, dropout_rate, class_size):
        """
        Text Parameters
        ---------------
        embedding_dim : int, default: 300
            The dimension of input vector
        window_sizes : tuple, default: (3,4,5)
        feature_map_size : int, default: 100
        dropout_rate : float, default: 0.5
        class_size : int, default: 48
            The number of classes
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.window_sizes = window_sizes
        self.feature_map_size = feature_map_size
        self.dropout_rate = dropout_rate
        self.class_size = class_size
        
        self.convs = nn.ModuleList([nn.Conv2d(1, self.feature_map_size, (window_size, self.embedding_dim)) 
                                    for window_size in self.window_sizes])
        self.linear = nn.Linear(self.feature_map_size * len(self.window_sizes), self.class_size)
        
    def forward(self, input):
        # input shape: (batch_size, 1, max_doc_len, embedding_dim) 
                
        feature_maps = [F.relu(conv(input)).squeeze(3) for conv in self.convs] 
        # feature_maps shape: (batch_size, feature_map_size, feature_map_dim)
        # feature_map_dim: max_doc_len - (window_size-1)
        
        max_pooled = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) 
                                   for feature_map in feature_maps] 
        # max_pooled shape: (batch_size, feature_map_size) 
        
        concatenated = torch.cat(max_pooled, 1) 
        # concatenated shape: (batch_size, feature_map_size)
        
        logits = self.linear(F.dropout(concatenated, self.dropout_rate))
        # output shape: (batch_size, class_size)
        
        return logits
    
    
# TextCNN layer + Dropout layer + Batch Normalization
class TextcnnRegularizedLayer(nn.Module):
    def __init__(self, embedding_dim, window_sizes, feature_map_size, dropout_rate, class_size):
        """
        Text Parameters
        ---------------
        embedding_dim : int, default: 300
            The dimension of input vector
        window_sizes : tuple, default: (3,4,5)
        feature_map_size : int, default: 100
        dropout_rate : float, default: 0.5
        class_size : int, default: 48
            The number of classes
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.window_sizes = window_sizes
        self.feature_map_size = feature_map_size
        self.dropout_rate = dropout_rate
        self.class_size = class_size
        
        self.convs = nn.ModuleList([nn.Conv2d(1, self.feature_map_size, (window_size, self.embedding_dim)) 
                                    for window_size in self.window_sizes])
        self.batch_norm = nn.BatchNorm1d(self.feature_map_size)
        self.linear = nn.Linear(self.feature_map_size * len(self.window_sizes), self.class_size)
        
    def forward(self, input):
        # input shape: (batch_size, 1, max_doc_len, embedding_dim) 
        input = F.dropout(input, self.dropout_rate)
                
        feature_maps = [conv(input).squeeze(3) for conv in self.convs] 
        # feature_maps shape: (batch_size, feature_map_size, feature_map_dim)
        # feature_map_dim: max_doc_len - (window_size-1)
        
        feature_maps_bned = [F.relu(self.batch_norm(feature_map)) 
                             for feature_map in feature_maps]
        # feature_maps_bned shape: (batch_size, feature_map_size, feature_map_dim)
        # feature_map_dim: max_doc_len - (window_size-1)
        
        max_pooled = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) 
                                   for feature_map in feature_maps_bned] 
        # max_pooled shape: (batch_size, feature_map_size)
        
        concatenated = torch.cat(max_pooled, 1) 
        # concatenated shape: (batch_size, feature_map_size)
        
        logits = self.linear(F.dropout(concatenated, self.dropout_rate))
        # output shape: (batch_size, class_size)
        
        return logits
    

# token index -> word vector / segment index -> segment vector / postion index -> position vector
class BertEmbeddingLayer(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    # !hidden_dropout_rate
    def __init__(self, vocab_size, max_sents_len=512, hidden_dim=768, hidden_dropout_rate=0.1, variance_epsilon=1e-12):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0) # token embedding
        self.pos_embed = nn.Embedding(max_sents_len, hidden_dim) # position embedding
        self.seg_embed = nn.Embedding(2, hidden_dim) # segment(token type) embedding
        
        self.norm = LayerNorm(hidden_dim, variance_epsilon)
        self.drop = nn.Dropout(hidden_dropout_rate)

    def forward(self, tokens_ids, segment_ids):
        # tokens_ids shape: (batch_size, max_sent_per_doc_len, max_sents_len)
        # segment_ids shape: (batch_size, max_sent_per_doc_len, max_sents_len)
        
        seq_len = tokens_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens_ids) # (S,) -> (B, S)

        e = self.tok_embed(tokens_ids) + self.pos_embed(position_ids) + self.seg_embed(segment_ids)
        return self.drop(self.norm(e))
    

# Self Attention 
class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, hidden_dim=768, attn_dropout_rate=0.1, n_heads=12):
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(attn_dropout_rate)
        self.scores = None # for visualization
        self.n_heads = n_heads

    def forward(self, x):
        # mask = None
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(hidden_dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(hidden_dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h
    
    
# Layer Normalization    
class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, hidden_dim=768, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta  = nn.Parameter(torch.zeros(hidden_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
    
    
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))
    
    
class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, hidden_dim=768, hidden_dropout_rate=0.1, 
                 attn_dropout_rate=0.1, n_heads=12, variance_epsilon=1e-12):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(hidden_dim, attn_dropout_rate, n_heads)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim, variance_epsilon)
        self.pwff = PositionWiseFeedForward(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim, variance_epsilon)
        self.drop = nn.Dropout(hidden_dropout_rate)

    def forward(self, x):
        h = self.attn(x)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h
    

# Transformer Layer
# Feature-based approach BERT
# n_top_layers만큼의 상위 n개 block들을 hidden state를 더하여 feature extraction
class TransformerLayer(nn.Module):
    """ 
    Transformer with Self-Attentive Blocks
    BERT feature-based approach
    """
    def __init__(self, vocab_size, n_top_layers=4, hidden_dim=768, max_sents_len=512, 
                 hidden_dropout_rate=0.1, attn_dropout_rate=0.1, n_heads=12, 
                 n_layers=12, variance_epsilon=1e-12):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_top_layers = n_top_layers
        
        self.embed = BertEmbeddingLayer(vocab_size, max_sents_len, hidden_dim, hidden_dropout_rate)
        self.blocks = nn.ModuleList([Block(hidden_dim, hidden_dropout_rate, 
                                           attn_dropout_rate, n_heads, variance_epsilon) 
                                     for _ in range(self.n_layers)])
        
    def forward(self, tokens_ids, segment_ids):
        
        h = self.embed(tokens_ids, segment_ids)
        
        features = None
        for layer_idx in range(self.n_layers):
            block = self.blocks[layer_idx]
            h = block(h)
            if layer_idx == (self.n_layers - (self.n_top_layers)):
                features = h
            elif layer_idx > (self.n_layers - (self.n_top_layers)):
                features += h
        return features
    
    
# Hierarchical Attention Network 참고    
class SentAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        linear = nn.Linear(hidden_dim, hidden_dim)
        dot_product_u_s = nn.Linear(hidden_dim,1,bias=False)
        
    def forward(self, s_i):
        # s_i shpae: (batch_size, cfg.max_sents_len, hidden_dim)
        
        u_i = F.tanh(self.linear(s_i)) 
        # context vector
        # u_i shape: (batch_size, cfg.max_sents_len, hidden_dim)
        
        a_i = F.softmax(self.dot_product_u_s(u_i))
        # a_i shape: (batch_size, cfg.max_sents_len, 1)
        
        v = (a_i.transpose(1,2) @ u_i).squeeze(1)
        # v shape: (batch_size, hidden_dim)
        

# word vector들 평균 -> sentence vector
class BertAverage(nn.Module):
    def __init__(self, vocab_size, n_top_layers=4, hidden_dim=768,
                 max_sents_len=512, hidden_dropout_rate=0.1, attn_dropout_rate=0.1,
                 n_heads=12, n_layers=12, variance_epsilon=1e-12):
        
        super().__init__()
        self.transformer = TransformerLayer(vocab_size, n_top_layers, hidden_dim,
                                            max_sents_len, hidden_dropout_rate, attn_dropout_rate,
                                            n_heads, n_layers, variance_epsilon)
        
    def forward(self, tokens_ids, segment_ids):
        # tokens_ids shape: (batch_size, max_seq_len)
        # segment_ids shape: (batch_size, max_seq_len)
        
        h = self.transformer(tokens_ids, segment_ids)
        # h shape: (batch_size, max_seq_len, hidden_dim)
#         s = h.mean(dim=1)    
        s = h[:,0] #['CLS'] vector만 추출   
    
        # s shape: (batch_size, hidden_dim)
        
        return s
    

# word vector들을 그대로 가져오기
class BertConcat(nn.Module):
    def __init__(self, vocab_size, n_top_layers=4, hidden_dim=768,
                 max_sents_len=512, hidden_dropout_rate=0.1, attn_dropout_rate=0.1,
                 n_heads=12, n_layers=12, variance_epsilon=1e-12):
        
        super().__init__()
        self.transformer = TransformerLayer(vocab_size, n_top_layers, hidden_dim,
                                            max_sents_len, hidden_dropout_rate, attn_dropout_rate,
                                            n_heads, n_layers, variance_epsilon)
        
    def forward(self, tokens_ids, segment_ids):
        # tokens_ids shape: (batch_size, max_seq_len)
        # segment_ids shape: (batch_size, max_seq_len)
        
        h = self.transformer(tokens_ids, segment_ids)
        # h shape: (batch_size, max_seq_len, hidden_dim)
        
        s = h.cat(dim=1)
        # s shape: (batch_size, hidden_dim)
        
        return s
    
    
    
# BERT 분류모형    
class BertAvgLinear(nn.Module):
    """
    BertAvgLinear Parameters
    -------------------
    embedding_dim : int, default: 768
        The dimension of input vector
    hidden_layer_dim : int, default: 1024
        The dimension of hidden layer
    hidden_layer_size : int, default: 2
        The ith element represents the number of neurons in the ith
        hidden layer.
    dropout_rate : float, default: 0.5
    class_size : int, default: 48
        The number of classes
    """
    def __init__(self, hidden_layer_size=2, embedding_dim=768, hidden_layer_dim=1024, dropout_rate=0.5, class_size=48):
        super().__init__()
        
        self.dropout_rate = dropout_rate 
        
        if hidden_layer_size < 1:
            raise ValueError()
            
        self.input_layer = nn.Linear(embedding_dim, hidden_layer_dim).double() # Input Layer
        
        hidden_layer_list = list()
        for i in range(hidden_layer_size-1): # Layer 수 
            hidden_layer_list += [nn.Dropout(dropout_rate), nn.Linear(hidden_layer_dim, hidden_layer_dim, nn.Tanh())]
        hidden_layer_list += [nn.Dropout(dropout_rate)]
        self.hidden_layers = nn.Sequential(*hidden_layer_list).double()
        
        self.output_layer = nn.Linear(hidden_layer_dim, class_size).double()
        
    def forward(self, x):
        
        h = F.tanh(self.input_layer(x)) # input layer
        h = self.hidden_layers(h) # hidden layer
        o = self.output_layer(h)
        
        return o