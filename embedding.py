"""
embedding.py
텍스트를 n차원 벡터로 임베딩시키는 클래스들을 정의한 모듈
"""


from utils import Instance, to_instances, Pipeline, set_seeds
from layers import BertAverage, BertConcat
import checkpoint

import os
import json
import pickle
from collections import namedtuple

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from tqdm import tqdm_notebook

import torch
from torch.nn import DataParallel

# Doc2vec Embedding
# Document Embedding
# gensim.models.doc2vec.Doc2vec을 사용
class D2vEmbedding(Pipeline):
    """
    D2vEmbedding Parameters
    -----------------------
    dm : {1,0}, optional, default: 0
        Defines the training algorithm. 
        If `dm=1`, 'distributed memory' (PV-DM) is used.
        Otherwise, `distributed bag of words` (PV-DBOW) is employed.
    embedding_dim : int, optional, default: 300
        Dimensionality of the feature vectors.
    window_size : int, optional, default: 15
        The maximum distance between the current and predicted word within a sentence.
    alpha : float, optional, default: 0.05
        The initial learning rate.
    min_alpha : float, optional, default: 0.005
        Learning rate will linearly drop to `min_alpha` as training progresses.
    sample : float, optional default: 1e-05
        The threshold for configuring which higher-frequency words are randomly downsampled,
        useful range is (0, 1e-5).
    workers : int, optional, default: 12
        Use these many worker threads to train the model (=faster training with multicore machines).
    negative : int, optional, default: 5
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    dbow_words : {1,0}, optional default: 1
        If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
        doc-vector training; If 0, only trains doc-vectors (faster).

        The input parameters are of the following types:
            * `word` (str) - the word we are examining
            * `count` (int) - the word's frequency count in the corpus
            * `min_count` (int) - the minimum count threshold.
    """
    def __init__(self, text_cnn, d2v_train, d2v_dir, d2v_model,
                 dm=0, embedding_dim=300, window_size=15, 
                 alpha=0.05, min_alpha=0.005,
                 sample=1e-05, workers=12, negative=5, 
                 dbow_words=1, epochs=30, seed=2019):
        super().__init__()
        
        self.text_cnn = text_cnn
        self.d2v_train = d2v_train
        self.d2v_dir = d2v_dir
        self.d2v_model = d2v_model
        self.d2v_path = os.path.join(d2v_dir, self.__class__.__name__+'_'+d2v_model +'.model')
        self.d2v_epochs = epochs
        
        if self.d2v_train:
            self.d2v = Doc2Vec(dm=dm, 
                               dbow_words=dbow_words, 
                               window=window_size,
                               vector_size=embedding_dim, 
                               alpha=alpha, 
                               min_alpha=min_alpha,
                               workers=workers, 
                               sample=sample, 
                               negative=negative,
                               seed=seed,
                               epochs=epochs)
            
        else:
            if os.path.isfile(self.d2v_path):
                self.d2v = Doc2Vec.load(self.d2v_path)
            else:
                raise FileNotFoundError()
    
    def make_tagged_documents(self, instances):
        tagged_documents = [TaggedDocument(text.split(' '), [label]) for i, (id_, text, label) in enumerate(instances)] # Doc2Vec Input 형식
        return tagged_documents
    
    def train(self, tagged_documents): # doc2vec 학습
        self.d2v.build_vocab(tagged_documents) # 단어 구축
        self.d2v.train(tagged_documents, total_examples=len(tagged_documents), epochs=self.d2v.epochs) # doc2vec train
        self.d2v.save(self.d2v_path) # doc2vec 저장
        
        # 6.21 TextCNN Supervised Doc2Vec 이용
        if self.text_cnn & self.d2v_train: # textcnn 사용할 때는 w2v model save
            with open(os.path.join(self.d2v_dir, self.d2v_model + 'w2v_index2word.json'), 'w') as f:
                json.dump(self.d2v.wv.index2word, f)
                
            with open(os.path.join(self.d2v_dir, self.d2v_model + 'w2v_lookup_table.pickle'), 'wb') as f:
                pickle.dump(self.d2v.wv.vectors, f)
  
        
    def infer_vector(self, document): # document vector 추출 함수
        self.d2v.random.seed(2019)
        vector = np.expand_dims(self.d2v.infer_vector(document), axis=0)
        return vector
        
    def __call__(self, instances):
        
        if self.d2v_train:
            tagged_documents = self.make_tagged_documents(instances) # tagged document로 형식 변환
            self.train(tagged_documents)
            
        instances_embedded = list()    
        
        for id_, text, label in instances:
            if self.text_cnn == False:
                text_embedded = self.infer_vector(text.split(' ')) # document to vector
                instances_embedded.append(Instance(id_, text_embedded, label)) 
                
            else:
                instances_embedded.append(Instance(id_, text, label))
            
        return instances_embedded
    

# Word2vec Embedding
# WordEmbedding
# gensim.models.Word2vec을 사용    
class W2vEmbedding(Pipeline):
    """
    Parameters
    ----------
    embedding_dim : int, default: 300
        Dimensionality of the word vectors.
    window_size : int, default: 15
        Maximum distance between the current and predicted word within a sentence.
    min_count : int, default: 1
        Ignores all words with total frequency lower than this.
    workers : int, default: 12
        Use these many worker threads to train the model (=faster training with multicore machines).
    negative : int, default: 5
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    alpha : float, default: 0.05
        The initial learning rate.
    min_alpha : float, default: 0.005
        Learning rate will linearly drop to `min_alpha` as training progresses.
    sample : float, default: 1e-5
        The threshold for configuring which higher-frequency words are randomly downsampled,
        useful range is (0, 1e-5).
    """
    def __init__(self, w2v_dir, embedding_dim=300, window_size=15, 
                 min_count=1, workers=12, negative=5, 
                 alpha=0.05, min_alpha=0.005, sample=1e-5):
        
        self.w2v_dir = w2v_dir
        self.w2v = Word2Vec(size=embedding_dim, 
                            window=window_size, 
                            min_count=min_count, 
                            sample=sample, 
                            workers=workers, 
                            sg=1, 
                            negative=negative)
        
    def _train(self, instances): # word2vec 학습
        documents = [instance.text.split(' ') for instance in instances]
        self.w2v.build_vocab(documents)
        self.w2v.train(documents, total_examples=len(documents), epochs=self.w2v.epochs)
        
    def _save_index2word(self): # word2vec index2word(vocabulary) 저장
        with open(os.path.join(self.w2v_dir,'w2v_index2word.json'), 'w') as f:
            json.dump(self.w2v.wv.index2word, f)
    
    def _save_lookup_table(self): # word2vec lookup_table(word vectors) 저장
        with open(os.path.join(self.w2v_dir, 'w2v_lookup_table.pickle'), 'wb') as f:
            pickle.dump(self.w2v.wv.vectors, f)
        
    def __call__(self, instances):
        self._train(instances)
        self._save_index2word()
        self._save_lookup_table()
        
        return instances
    

# BertEmbedding    
# Sentence Embedding
# layers.py의 BertAverage를 사용
class BertEmbedding(Pipeline):
    def __init__(self, vocab_size, pretrain_file, device, data_parallel, n_top_layers=4, hidden_dim=768,
                 max_sents_len=512, hidden_dropout_rate=0.1, attn_dropout_rate=0.1,
                 n_heads=12, n_layers=12, variance_epsilon=1e-12):
        
        super().__init__()
        self.pretrain_file = pretrain_file
        self.data_parallel = data_parallel
        self.device = device
        
        set_seeds(42)
        
        self.model = BertAverage(vocab_size, n_top_layers, hidden_dim,
                                 max_sents_len, hidden_dropout_rate, attn_dropout_rate,
                                 n_heads, n_layers, variance_epsilon)
        
        self.model.eval()
        self.load(self.model, self.pretrain_file)
        self.model.double().to(self.device)
        if self.data_parallel:
            self.model = DataParallel(self.model)
            
        self.Instance = namedtuple('Instance', ['id', 'sentence', 'label'])
        
    def load(self, model, pretrain_file): # pre-trained model loading
        
        print('Loading the pretrained model from', pretrain_file)
        
        if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
            checkpoint.load_model(model.transformer, pretrain_file)
        else:
            model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )
        
    def __call__(self, data_loader):
        
        batch_size = data_loader.batch_size
        with torch.no_grad():
            ids_list = None
            tokens_ids_list = None
            segment_ids_list = None
            labels_list = None
            pbar = tqdm_notebook(data_loader)
            for i, batch in enumerate(pbar):
                ids, tokens_ids, segment_ids, labels = batch
                sentences = self.model(tokens_ids, segment_ids)
                if i == 0:
                    ids_list = ids
                    sentences_list = list(sentences)
                    labels_list = list(labels)
                else:
                    ids_list += ids
                    sentences_list += list(sentences)
                    labels_list += list(labels)
                pbar.set_description(f"Sentence {(i+1)*batch_size}" )
                    
        instances = [self.Instance(id_, sentence, label) 
                     for id_, sentence, label 
                     in zip(ids_list, sentences_list, labels_list)]

        return instances

# BertConcat을 이용해보기

class BertEmbedding_Concat(Pipeline):
    def __init__(self, vocab_size, pretrain_file, device, data_parallel, n_top_layers=4, hidden_dim=768,
                 max_sents_len=512, hidden_dropout_rate=0.1, attn_dropout_rate=0.1,
                 n_heads=12, n_layers=12, variance_epsilon=1e-12):
        
        super().__init__()
        self.pretrain_file = pretrain_file
        self.data_parallel = data_parallel
        self.device = device
        
        self.model = BertConcat(vocab_size, n_top_layers, hidden_dim,
                                 max_sents_len, hidden_dropout_rate, attn_dropout_rate,
                                 n_heads, n_layers, variance_epsilon)
        
        self.model.eval()
        self.load(self.model, self.pretrain_file)
        self.model.double().to(self.device)
        if self.data_parallel:
            self.model = DataParallel(self.model)
            
        self.Instance = namedtuple('Instance', ['id', 'sentence', 'label'])
        
    def load(self, model, pretrain_file): # pre-trained model loading
        
        print('Loading the pretrained model from', pretrain_file)
        
        if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
            checkpoint.load_model(model.transformer, pretrain_file)
        
    def __call__(self, data_loader):
        
        batch_size = data_loader.batch_size
        with torch.no_grad():
            ids_list = None
            tokens_ids_list = None
            segment_ids_list = None
            labels_list = None
            pbar = tqdm_notebook(data_loader)
            for i, batch in enumerate(pbar):
                ids, tokens_ids, segment_ids, labels = batch
                sentences = self.model(tokens_ids, segment_ids)
                if i == 0:
                    ids_list = ids
                    sentences_list = list(sentences)
                    labels_list = list(labels)
                else:
                    ids_list += ids
                    sentences_list += list(sentences)
                    labels_list += list(labels)
                pbar.set_description(f"Sentence {(i+1)*batch_size}" )
                    
        instances = [self.Instance(id_, sentence, label) 
                     for id_, sentence, label 
                     in zip(ids_list, sentences_list, labels_list)]

        return instances