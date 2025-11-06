"""
MIND 데이터셋을 LLaRA 프로젝트 형식으로 전처리하는 스크립트
"""
import pandas as pd
import numpy as np
import pickle as pkl
import os
import os.path as op
from collections import defaultdict

def load_news_mapping(news_file='MIND_news.tsv'):
    """뉴스 ID를 뉴스 제목으로 매핑하는 딕셔너리 생성"""
    print("Loading news mapping...")
    news_df = pd.read_csv(news_file, sep='\t', header=None)
    # 컬럼: 0=news_id, 1=category, 2=subcategory, 3=title, 4=body
    news_id2name = {}
    news_id2idx = {}
    
    for idx, row in news_df.iterrows():
        news_id_str = str(row[0])  # N1, N2, ...
        news_id_num = int(news_id_str[1:])  # 숫자 부분만 추출
        news_title = str(row[3]) if pd.notna(row[3]) else "Unknown"
        news_id2name[news_id_num] = news_title
        news_id2idx[news_id_str] = news_id_num
    
    print(f"Loaded {len(news_id2name)} news items")
    return news_id2name, news_id2idx

def parse_news_id(news_id_str):
    """N123 형식의 뉴스 ID를 숫자로 변환"""
    if isinstance(news_id_str, str) and news_id_str.startswith('N'):
        return int(news_id_str[1:])
    return int(news_id_str)

def create_sequences(mind_file='MIND.tsv', news_id2name=None, padding_item_id=None):
    """MIND.tsv를 읽어서 시퀀스 데이터 생성"""
    print("Loading MIND.tsv...")
    df = pd.read_csv(mind_file, sep='\t', header=None)
    
    # 모든 뉴스 ID 수집하여 padding_item_id 결정
    all_news_ids = set()
    for idx, row in df.iterrows():
        # sequence에서 뉴스 ID 추출
        seq_str = str(row[1])
        seq_ids = [parse_news_id(nid) for nid in seq_str.split()]
        all_news_ids.update(seq_ids)
        
        # groundtruth에서 뉴스 ID 추출
        gt_str = str(row[2])
        gt_ids = [parse_news_id(nid) for nid in gt_str.split()]
        all_news_ids.update(gt_ids)
    
    if padding_item_id is None:
        padding_item_id = max(all_news_ids) + 1
    
    print(f"Padding item ID: {padding_item_id}")
    print(f"Total unique news items: {len(all_news_ids)}")
    
    # 시퀀스 데이터 생성
    session_data = []
    
    for idx, row in df.iterrows():
        user_id = int(row[0])
        seq_str = str(row[1])
        gt_str = str(row[2])
        
        # 시퀀스 파싱
        seq_ids = [parse_news_id(nid) for nid in seq_str.split()]
        if len(seq_ids) < 3:  # 최소 길이 체크
            continue
        
        # Groundtruth 파싱 (세 번째 컬럼의 모든 항목을 후보로 사용, 첫 번째가 정답)
        gt_ids = [parse_news_id(nid) for nid in gt_str.split()]
        if not gt_ids:
            continue
        
        next_item = gt_ids[0]  # 첫 번째가 정답
        candidates = gt_ids  # 모든 항목을 후보로 사용 (보통 5개)
        
        # 모든 후보가 news_id2name에 있는지 확인
        if next_item not in news_id2name:
            continue
        if not all(cid in news_id2name for cid in candidates):
            continue
        
        # 패딩 추가 (최대 길이 50으로 제한)
        max_len = 50
        seq_padded = seq_ids[-max_len:] + [padding_item_id] * max(0, max_len - len(seq_ids))
        len_seq = min(len(seq_ids), max_len)
        
        session_data.append({
            'user_id': user_id,
            'seq': seq_padded,
            'seq_unpad': seq_ids[-max_len:],  # 최근 max_len개만 사용
            'len_seq': len_seq,
            'next': next_item,
            'candidates': candidates  # 세 번째 컬럼의 모든 항목 저장
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Created {len(session_df)} sessions")
    return session_df, padding_item_id

def split_data(session_df, train_ratio=0.7, val_ratio=0.15):
    """데이터를 train/val/test로 분할"""
    print("Splitting data...")
    n_total = len(session_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 랜덤 셔플
    session_df = session_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = session_df[:n_train].copy()
    val_df = session_df[n_train:n_train+n_val].copy()
    test_df = session_df[n_train+n_val:].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def save_dataframes(train_df, val_df, test_df, news_id2name, output_dir='data/ref/mind'):
    """DataFrame을 pickle로 저장하고 id2name.txt 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame 저장
    train_df.to_pickle(op.join(output_dir, 'train_data.df'))
    val_df.to_pickle(op.join(output_dir, 'Val_data.df'))
    test_df.to_pickle(op.join(output_dir, 'Test_data.df'))
    print(f"Saved DataFrames to {output_dir}")
    
    # id2name.txt 저장
    id2name_path = op.join(output_dir, 'id2name.txt')
    with open(id2name_path, 'w', encoding='utf-8') as f:
        for news_id, news_name in sorted(news_id2name.items()):
            # 탭과 줄바꿈 제거
            news_name_clean = news_name.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{news_id}::{news_name_clean}\n")
    print(f"Saved id2name.txt to {id2name_path}")

def main():
    # 뉴스 매핑 로드
    news_id2name, news_id2idx = load_news_mapping('MIND_news.tsv')
    
    # 시퀀스 데이터 생성
    session_df, padding_item_id = create_sequences('MIND.tsv', news_id2name)
    
    # 데이터 분할
    train_df, val_df, test_df = split_data(session_df)
    
    # 저장
    save_dataframes(train_df, val_df, test_df, news_id2name, 'data/ref/mind')
    
    print(f"\nPreprocessing complete!")
    print(f"Padding item ID: {padding_item_id}")
    print(f"Output directory: data/ref/mind")

if __name__ == '__main__':
    main()

