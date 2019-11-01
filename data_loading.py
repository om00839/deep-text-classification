"""
data_loading.py
데이터를 torch.utils.data.DataLoader 타입으로 만들어주기 위한 모듈
"""

from tokenizing import load_vocab

from utils import Pipeline

from functools import partial
from collections import namedtuple
from itertools import groupby
from operator import attrgetter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# DataLoading에 공통적으로 사용되는 method들을 정의한 추상클래스
class BaseDataLoading(Dataset, Pipeline):
    def __init__(self, batch_size, device):
        super(BaseDataLoading, self).__init__()
        self.batch_size = batch_size
        self.device = device
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        text = self.instances[idx].text
        label = self.instances[idx].label
        return text, label

    
# D2vMlp 기반 모델들에 공통적으로 사용되는 DataLoading 클래스

class D2vMlpDataLoading(BaseDataLoading):
    def __init__(self, batch_size, device):
        super().__init__(batch_size, device)
    
    # collate_fn: batch_size만큼 데이터를 합치는데 사용되는 함수
    def _d2v_mlp_collate(self, batch):
        ids = list()
        texts = list()
        labels = list()
        for id_, text, label in batch:
            ids += [id_]
            texts += [text]#*length # CrossEntropy를 위해 수정 190704
            labels += [label]
        texts = torch.DoubleTensor(np.concatenate(texts, axis=0)).to(self.device) # CrossEntropy를 위해 수정 190704
        labels = torch.DoubleTensor(labels).long().to(self.device) # CrossEntropy를 위해 수정 190704
#         texts = torch.DoubleTensor(np.concatenate(texts, axis=0)).to(self.device)
#         labels = torch.DoubleTensor(np.concatenate(labels, axis=0)).to(self.device)
        return ids, texts, labels
        
    def __call__(self, instances):
        
        self.instances = instances
        
        return DataLoader(self.instances, batch_size=self.batch_size, collate_fn=self._d2v_mlp_collate) 
    
    
# TextCNN 기반 모델에 공통적으로 사용되는 DataLoading 클래스
class TextcnnDataLoading(BaseDataLoading):
    def __init__(self, batch_size, device):
        super().__init__(batch_size, device)

    # collate_fn: batch_size만큼 데이터를 합치는데 사용되는 함수    
    def _textcnn_collate(self, batch):
        ids = list()
        texts = list()
        labels = list()
        for id_, text, label in batch:
            ids += [id_]
#             length = label.shape[0]
            texts += [text]#*length
            labels += [label]
        labels = torch.DoubleTensor(labels).long().to(self.device) # CrossEntropy를 위해 수정 190704
#         labels = torch.DoubleTensor(np.concatenate(labels, axis=0)).to(self.device)
        return ids, texts, labels

    def __call__(self, instances):

        self.instances = instances
        dataloader = DataLoader(self.instances, batch_size=self.batch_size, collate_fn=self._textcnn_collate)
        return dataloader
    

# BertEmbedding에 사용되는 DataLoading 클래스    
class BertDataLoading(BaseDataLoading):
    
    def __init__(self, vocab_file, batch_size, device):
        
        super().__init__(batch_size, device)
        self.vocab = load_vocab(vocab_file)
        self.MAX_SEQ_LEN = 512
        self.Instance = namedtuple('Instance', ['id', 'sent_len', 'tokens_id', 'segment_id', 'label'])
    
    # collate_fn: batch_size만큼 데이터를 합치는데 사용되는 함수
    def _bert_collate(self, batch):
        
        ids = list()
        tokens_ids = list()
        segment_ids = list()
        labels = list()
        for id_, sent_len, tokens_id, segment_id, label in batch:
            ids.append(id_)
            tokens_ids.append(tokens_id)
            segment_ids.append(segment_id)
            labels.append(label)
        tokens_ids = torch.LongTensor(tokens_ids).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).to(self.device)
        labels = torch.LongTensor(labels).to(self.device) # Cross Entropy를 위해 190710 수정
        
        return ids, tokens_ids, segment_ids, labels
    
    def _add_special_words_with_truncation(self, words):
        # words: list of str
        # -2 special words for [CLS] text_a [SEP]
        _max_len = self.MAX_SEQ_LEN - 2
        
        if len(words) > _max_len:
            words = words[:_max_len]
            
        # Add Special words
        words = ['[CLS]'] + words + ['[SEP]']
        
        return words
                
    def _token_indexing(self, words):
        
        indices = []
        for token in words:
            token = token.strip()
            if token == '':
                continue
            indices.append(self.vocab[token])
            
        return indices
    
    def _add_words_zero_padding(self, words):
        
        n_pad = self.MAX_SEQ_LEN - len(words)
        words += [0]*n_pad
        
        return words
    
    def get_max_sent_per_doc_len(self):
        return self.max_sent_per_doc_len
    
    def __call__(self, instances):
        max_sent_per_doc_len = 0
        self.instances = list()
        for id_, text, label in instances:
            sents = text.split('\n') # list of str
            sent_len = 0
            for sent in sents:
                sent_len += 1
                words = sent.split(' ')
                words = map(lambda x: x.strip(), words)
                words = filter(lambda x: True if x !='' else False, words)
                words = list(words)
                words = self._add_special_words_with_truncation(words)
                tokens_id = self._token_indexing(words)
                tokens_id = self._add_words_zero_padding(tokens_id)
                segment_id = [0]*self.MAX_SEQ_LEN
                self.instances.append(self.Instance(id_, sent_len, tokens_id, segment_id, label))
            self.max_sent_per_doc_len = max(max_sent_per_doc_len, sent_len)
            
        return DataLoader(self.instances, batch_size=self.batch_size, collate_fn=self._bert_collate)
    

# BertDAP 기반 모델에 공통적으로 사용되는 DataLoading 클래스
# BertDAP(BERT Document Average Pooling): 
# BertEmbedding을 통해 얻은 Sentence Vector들의 평균을 통해 Document Vector를 계산
class BertDAPDataLoading(BaseDataLoading):
    def __init__(self, batch_size, device):
        super().__init__(batch_size, device)
        
        self.Instance = namedtuple('Instance', ['id_', 'document', 'label'])
        self.instances = list()
        
    # class Dataset - method override
    def __len__(self):
        return len(self.instances)
    
    # class Dataset - method override
    def __getitem__(self, idx):
        return self.instances[idx]
    
    # collate_fn: batch_size만큼 데이터를 합치는데 사용되는 함수
    def _collate(self, batch):
        
        ids = list()
        documents = list()
        labels = list()
        
        for id_, document, label in batch:
            ids.append(id_)
            documents.append(document.unsqueeze(0))
            labels.append(label)
    
        ids = ids
        documents = torch.cat(documents)
#         labels = torch.cat(labels)
        labels = torch.LongTensor(labels).to(self.device) # 190710 Cross Entropy로 수정
        
        return ids, documents, labels
        
    # class Pipeline - method override
    def __call__(self, instances):
        
        for id_, instances in groupby(instances, key=attrgetter('id')):
            document = None
            N = 0
            for _, sentence, label in instances:
                N += 1
                if document is None:
                    document = sentence
                else:
                    document += sentence
                document = document / N
            self.instances.append(self.Instance(id_, document, label))
                
        return DataLoader(self.instances, batch_size=self.batch_size, collate_fn = self._collate)