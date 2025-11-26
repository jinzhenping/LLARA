"""
MIND 데이터셋 (behaviors_new.tsv, news.tsv)을 LLaRA 프로젝트 형식으로 전처리하는 스크립트
"""
import pandas as pd
import numpy as np
import pickle as pkl
import os
import os.path as op
from collections import defaultdict
import random

def load_news_mapping(news_file='mind_new/news.tsv', behaviors_file=None):
    """뉴스 ID를 뉴스 제목으로 매핑하는 딕셔너리 생성"""
    print("Loading news mapping...")
    news_df = pd.read_csv(news_file, sep='\t', header=None, on_bad_lines='skip')
    
    # news.tsv 형식: news_id, category, subcategory, title, abstract, url, ...
    news_id2name = {}
    news_id2idx = {}
    
    for idx, row in news_df.iterrows():
        try:
            news_id_str = str(row[0]).strip()  # N88753 형식
            if not news_id_str.startswith('N'):
                continue
                
            news_id_num = int(news_id_str[1:])  # 숫자 부분만 추출
            
            # 카테고리, 서브카테고리, 제목 추출
            category = str(row[1]) if len(row) > 1 and pd.notna(row[1]) else "Unknown"
            subcategory = str(row[2]) if len(row) > 2 and pd.notna(row[2]) else "Unknown"
            title = str(row[3]) if len(row) > 3 and pd.notna(row[3]) else "Unknown"
            
            # 제목이 비어있으면 abstract 사용
            if title == "Unknown" or title == "":
                title = str(row[4]) if len(row) > 4 and pd.notna(row[4]) else "Unknown"
            
            # 형식: "카테고리 - 서브카테고리: 제목" (제목만 사용할 수도 있음)
            # 제목이 너무 길면 잘라냄
            if len(title) > 200:
                title = title[:200] + "..."
            
            news_name = f"{category} - {subcategory}: {title}"
            news_id2name[news_id_num] = news_name
            news_id2idx[news_id_str] = news_id_num
        except Exception as e:
            continue
    
    print(f"Loaded {len(news_id2name)} news items")
    
    # behaviors_file이 제공된 경우, 사용되는 뉴스 ID 정보 출력
    if behaviors_file:
        used_news_ids = set()
        try:
            with open(behaviors_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        seq_str = parts[1].strip()
                        for nid_str in seq_str.split():
                            nid = parse_news_id(nid_str.strip())
                            if nid is not None:
                                used_news_ids.add(nid)
            matched = len(set(news_id2name.keys()) & used_news_ids)
            print(f"Found {len(used_news_ids)} unique news IDs in behaviors file")
            print(f"Matched {matched} news items that are used in behaviors file")
        except Exception as e:
            print(f"Warning: Could not scan behaviors file: {e}")
    
    return news_id2name, news_id2idx

def parse_news_id(news_id_str):
    """N123 형식의 뉴스 ID를 숫자로 변환"""
    if isinstance(news_id_str, str) and news_id_str.startswith('N'):
        try:
            return int(news_id_str[1:])
        except:
            return None
    try:
        return int(news_id_str)
    except:
        return None

def create_sequences(behaviors_file='mind_new/behaviors_194_users.tsv', news_id2name=None, padding_item_id=None, min_seq_len=3, max_seq_len=50):
    """behaviors_194_users.tsv를 읽어서 시퀀스 데이터 생성 (첫 번째, 두 번째 컬럼만 사용)"""
    print(f"Loading {behaviors_file}...")
    print("Using only first and second columns (user_id and history sequence)")
    
    # 모든 뉴스 ID 수집하여 padding_item_id 결정
    all_news_ids = set()
    session_data_raw = []
    
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 1000 == 0 and line_idx > 0:
                print(f"Processing line {line_idx}...")
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            user_id = parts[0].strip()
            seq_str = parts[1].strip()  # 두 번째 컬럼: 공백으로 구분된 뉴스 ID 리스트 (히스토리)
            
            # 시퀀스 파싱 (두 번째 컬럼만 사용)
            seq_ids = []
            for nid_str in seq_str.split():
                nid = parse_news_id(nid_str.strip())
                if nid is not None:
                    seq_ids.append(nid)
                    all_news_ids.add(nid)
            
            if len(seq_ids) < min_seq_len:
                continue
            
            session_data_raw.append({
                'user_id': user_id,
                'seq_ids': seq_ids
            })
    
    if padding_item_id is None:
        padding_item_id = max(all_news_ids) + 1 if all_news_ids else 130319
    
    print(f"Padding item ID: {padding_item_id}")
    print(f"Total unique news items: {len(all_news_ids)}")
    print(f"Total sessions: {len(session_data_raw)}")
    
    # 시퀀스 데이터 생성 (마지막 아이템을 정답으로, 나머지를 히스토리로)
    session_data = []
    
    for session in session_data_raw:
        seq_ids = session['seq_ids']
        
        if len(seq_ids) < min_seq_len:
            continue
        
        # 마지막 아이템을 정답으로, 나머지를 히스토리로
        history = seq_ids[:-1]
        next_item = seq_ids[-1]
        
        # 모든 아이템이 news_id2name에 있는지 확인
        if next_item not in news_id2name:
            continue
        if not all(nid in news_id2name for nid in history):
            continue
        
        # 패딩 추가 (최대 길이 제한)
        history_padded = history[-max_seq_len:] + [padding_item_id] * max(0, max_seq_len - len(history))
        len_seq = min(len(history), max_seq_len)
        
        # 후보 생성: 정답 + 9개 랜덤 후보 (나중에 negative sampling으로 대체될 수 있음)
        # 여기서는 일단 정답만 저장하고, 나중에 negative sampling 수행
        candidates = [next_item]  # 일단 정답만, 나중에 negative sampling
        
        session_data.append({
            'user_id': session['user_id'],
            'seq': history_padded,
            'seq_unpad': history[-max_seq_len:],  # 최근 max_seq_len개만 사용
            'len_seq': len_seq,
            'next': next_item,
            'candidates': candidates  # 나중에 negative sampling으로 채워짐
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Created {len(session_df)} sessions")
    return session_df, padding_item_id, all_news_ids

def add_negative_samples(session_df, news_id2name, cans_num=10):
    """각 세션에 negative sampling으로 후보 추가"""
    print("Adding negative samples...")
    
    all_item_ids = list(news_id2name.keys())
    
    def sample_candidates(row):
        seq_unpad = row['seq_unpad']
        next_item = row['next']
        
        # 후보 풀: 히스토리에 없고 정답이 아닌 아이템들
        candidate_pool = [item_id for item_id in all_item_ids 
                         if item_id not in seq_unpad and item_id != next_item]
        
        # negative sampling
        if len(candidate_pool) >= cans_num - 1:
            negatives = random.sample(candidate_pool, cans_num - 1)
        else:
            negatives = candidate_pool
        
        # 정답 + negative 샘플들
        candidates = negatives + [next_item]
        random.shuffle(candidates)  # 정답 위치 숨기기
        
        return candidates
    
    session_df['candidates'] = session_df.apply(sample_candidates, axis=1)
    print("Negative sampling complete")
    return session_df

def split_data(session_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """데이터를 train/val/test로 분할"""
    print("Splitting data...")
    n_total = len(session_df)
    
    # 비율 정규화
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 랜덤 셔플
    session_df = session_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = session_df[:n_train].copy()
    val_df = session_df[n_train:n_train+n_val].copy()
    test_df = session_df[n_train+n_val:].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def save_dataframes(train_df, val_df, test_df, news_id2name, output_dir='data/ref/mind_new'):
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
            news_name_clean = news_name.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{news_id}::{news_name_clean}\n")
    print(f"Saved id2name.txt to {id2name_path}")

def main():
    # 파일 경로 설정
    behaviors_file = 'mind_new/behaviors_194_users.tsv'  # 첫 번째, 두 번째 컬럼만 사용
    news_file = 'mind_new/news.tsv'
    output_dir = 'data/ref/mind_194_users'
    
    # 뉴스 매핑 로드 (behaviors_file 정보도 전달하여 사용되는 ID만 로드)
    news_id2name, news_id2idx = load_news_mapping(news_file, behaviors_file)
    
    if len(news_id2name) == 0:
        print("Error: No news items loaded. Check the news.tsv file format.")
        return
    
    # 시퀀스 데이터 생성
    session_df, padding_item_id, all_news_ids = create_sequences(
        behaviors_file, 
        news_id2name,
        padding_item_id=None,
        min_seq_len=3,
        max_seq_len=50
    )
    
    if len(session_df) == 0:
        print("Error: No sessions created. Check the behaviors_new.tsv file format.")
        return
    
    # Negative sampling 추가
    session_df = add_negative_samples(session_df, news_id2name, cans_num=10)
    
    # 데이터 분할
    train_df, val_df, test_df = split_data(session_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 저장
    save_dataframes(train_df, val_df, test_df, news_id2name, output_dir)
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"{'='*50}")
    print(f"Padding item ID: {padding_item_id}")
    print(f"Total news items: {len(news_id2name)}")
    print(f"Total sessions: {len(session_df)}")
    print(f"Train sessions: {len(train_df)}")
    print(f"Val sessions: {len(val_df)}")
    print(f"Test sessions: {len(test_df)}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Update train_mind.sh to use data_dir='{output_dir}'")
    print(f"2. Update padding_item_id in main.py if needed (current: {padding_item_id})")
    print(f"3. Run: sh train_mind.sh")

if __name__ == '__main__':
    main()

