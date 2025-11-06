import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random

class MindData(data.Dataset):
    def __init__(self, data_dir=r'data/ref/mind',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=130319
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        next_item = temp['next']
        
        # 세 번째 컬럼에 저장된 후보 사용 (negative sampling 대신)
        # 원본 순서 그대로 사용 (첫 번째 항목이 정답)
        if 'candidates' in temp.index and isinstance(temp['candidates'], list) and len(temp['candidates']) > 0:
            candidates = temp['candidates'].copy()
        else:
            # fallback: 기존 방식 (데이터에 candidates가 없는 경우)
            candidates = self.negative_sampling(temp['seq_unpad'], next_item)
        
        cans_name=[self.item_id2name.get(can, "Unknown") for can in candidates]
        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': len(candidates),  # 동적으로 설정
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        # fallback용 (사용되지 않을 예정)
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_news_id2name()
        if self.stage=='train':
            filename="train_data.df"
        elif self.stage=='val':
            filename="Val_data.df"
        elif self.stage=='test':
            filename="Test_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  
    
    def get_news_id2name(self):
        news_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                if len(ll) >= 2:
                    news_id2name[int(ll[0])] = ll[1].strip()
        return news_id2name
    
    def session_data4frame(self, datapath, news_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        def seq_to_title(x): 
            return [news_id2name.get(x_i, "Unknown") for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return news_id2name.get(x, "Unknown")
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        
        # candidates가 리스트로 저장되어 있는지 확인하고 처리
        if 'candidates' in train_data.columns:
            # candidates가 이미 리스트인 경우 그대로 사용
            pass
        else:
            # candidates가 없는 경우 빈 리스트로 초기화 (이 경우는 없어야 함)
            train_data['candidates'] = [[] for _ in range(len(train_data))]
        
        return train_data

