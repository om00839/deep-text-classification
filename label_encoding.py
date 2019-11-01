"""
label_encoding.py
라벨의 형태를 학습의 loss function에 맞게 변환하기 위한 모듈
"""


from utils import Instance, to_instances, Pipeline
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

import json
import numpy as np


class BaseLabelEncoding(Pipeline):
    def __init__(self, cls_to_idx_path='./cls_to_idx.json'):
        super().__init__()
        
        # cls_to_idx(class to index dictionary)
        # {class_name:class_index, ...}
        # class index를 고정시켜두기 위해 json 파일로 저장 후 사용
        with open(cls_to_idx_path, 'r') as f:
            self.cls_to_idx = json.load(f)


# n개 클래스에 대해서 one vs. rest classification
# Loss Function: MultiLabelSoftMargin(or BCEWithLogitsLoss) 사용
class BinaryLabelEncoding(BaseLabelEncoding):
    
    """
    LabelBinaryEncodig: 해당 label이 있을경우 1, 없을 경우 0으로 인코딩 
    
    예시)
    Input
    ID   | Label 
    1    | label1, label6, ...        
    
    LabelBinaryEncoding
    ID   | Label1 | Label2 | ... | Label6 | ... | LabelN
    1    | 1      | 0      |     | 1      |     | 0  
    """
    
    def __init__(self, cls_to_idx_path='./cls_to_idx.json'):
        super().__init__(cls_to_idx_path)
        
    def transform(self, label):
        
        label_encoded = np.zeros((1,len(self.cls_to_idx)))
        
        for cls in label.split('\n'):
            col = self.cls_to_idx[cls]
            label_encoded[0,col] = 1
            
        return label_encoded
        
    def __call__(self, instances):
        
        instances_encoded = list()
        
        for id_, text, label in instances:
            label_encoded = self.transform(label)
            instances_encoded.append(Instance(id_, text, label_encoded))
            
        return instances_encoded
        

# 각각의 label에 대해 multi-class classification        
# Loss Function: CrossEntropyLoss 사용
class SingleLabelEncoding(BaseLabelEncoding):
    
    """
    LabelSingleEncoding: 하나의 label을 하나의 행으로 인코딩
    
    예시)
    Input
    ID   | Label 
    1    | label1, label6, ...        
    
    LabelSingleEncoding
    ID   | Label
    1    | label1
    1    | label6
    """
    
    def __init__(self, cls_to_idx_path='./cls_to_idx.json'):
        super().__init__(cls_to_idx_path)
        
    def transform(self, label):
        
#         clss = label.split('\n')
#         label_encoded = np.zeros((len(clss),1))
        
#         for row, cls in enumerate(clss):
        idx = self.cls_to_idx[label]
#         label_encoded[row,0] = idx
            
        return idx #,label_encoded
        
    def __call__(self, instances):
        
#         self.fit(instances)
        
        instances_encoded = list()
        
        for id_, text, label in instances:
            label_encoded = self.transform(label)
            instances_encoded.append(Instance(id_, text, label_encoded))
            
        return instances_encoded


class LabelDecoding(BaseLabelEncoding):
    
    """
    LabelSingleEncoding: 하나의 label을 하나의 행으로 인코딩
    
    예시)
    Input
    ID   | Label 
    1    | label1, label6, ...        
    
    LabelSingleEncoding
    ID   | Label
    1    | label1
    1    | label6
    """
    
    def __init__(self, cls_to_idx_path='./cls_to_idx.json'):
        super().__init__(cls_to_idx_path)
        
    def transform(self, label):
        
        pred = [list(self.cls_to_idx.keys())[list(self.cls_to_idx.values()).index(x)] for x in label]
        
        return pred 