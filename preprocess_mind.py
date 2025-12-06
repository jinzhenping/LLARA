"""
MIND 데이터셋을 LLaRA 프로젝트 형식으로 전처리하는 스크립트
"""
import pandas as pd
import numpy as np
import pickle as pkl
import os
import os.path as op
from collections import defaultdict
import random

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
        
        # 카테고리, 서브카테고리, 제목을 모두 포함
        category = str(row[1]) if pd.notna(row[1]) else "Unknown"
        subcategory = str(row[2]) if pd.notna(row[2]) else "Unknown"
        title = str(row[3]) if pd.notna(row[3]) else "Unknown"
        
        # 형식: "카테고리 - 서브카테고리: 제목"
        news_name = f"{category} - {subcategory}: {title}"
        news_id2name[news_id_num] = news_name
        news_id2idx[news_id_str] = news_id_num
    
    print(f"Loaded {len(news_id2name)} news items")
    return news_id2name, news_id2idx

def parse_news_id(news_id_str):
    """N123 형식의 뉴스 ID를 숫자로 변환"""
    if isinstance(news_id_str, str) and news_id_str.startswith('N'):
        return int(news_id_str[1:])
    return int(news_id_str)

def create_sequences(mind_file='MIND.tsv', news_id2name=None, padding_item_id=None):
    """MIND.tsv를 읽어서 시퀀스 데이터 생성
    - 학습/검증용: 두 번째 컬럼(히스토리)에서 마지막 아이템을 정답으로 사용
    - 테스트용: 세 번째 컬럼(후보)의 데이터 사용
    """
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
        if pd.notna(gt_str) and str(gt_str).strip():
            gt_ids = [parse_news_id(nid) for nid in str(gt_str).split()]
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
        gt_str = str(row[2]) if pd.notna(row[2]) else ""
        
        # 시퀀스 파싱
        seq_ids = [parse_news_id(nid) for nid in seq_str.split()]
        if len(seq_ids) < 3:  # 최소 길이 체크
            continue
        
        # 세 번째 컬럼(후보) 파싱
        gt_ids = []
        if gt_str and str(gt_str).strip():
            gt_ids = [parse_news_id(nid) for nid in str(gt_str).split()]
        
        # 패딩 추가 (최대 길이 50으로 제한)
        max_len = 50
        
        # 학습/검증용: 두 번째 컬럼(히스토리)에서 마지막 아이템을 정답으로
        # 테스트용: 두 번째 컬럼의 모든 데이터를 히스토리로 사용 (정답은 세 번째 컬럼에서)
        history_for_train = seq_ids[:-1]  # 학습/검증용: 마지막 아이템 제외한 히스토리
        history_for_test = seq_ids  # 테스트용: 전체 히스토리 (마지막 아이템 포함)
        next_item_from_history = seq_ids[-1]  # 마지막 아이템이 정답 (학습/검증용)
        
        if next_item_from_history not in news_id2name:
            continue
        
        # 테스트용: 세 번째 컬럼의 후보 사용 (있는 경우)
        if gt_ids and len(gt_ids) > 0:
            # 세 번째 컬럼이 있으면 후보로 사용
            next_item = gt_ids[0]  # 첫 번째가 정답
            candidates = gt_ids  # 모든 항목을 후보로 사용
            
            # 모든 후보가 news_id2name에 있는지 확인
            if next_item not in news_id2name:
                continue
            if not all(cid in news_id2name for cid in candidates):
                continue
            
            # 테스트용 히스토리 사용 (전체 포함)
            history = history_for_test
        else:
            # 세 번째 컬럼이 없으면 학습/검증용으로만 사용
            next_item = next_item_from_history
            candidates = []  # 후보 없음 (학습/검증용)
            # 학습/검증용 히스토리 사용 (마지막 제외)
            history = history_for_train
        
        # 히스토리 패딩
        history_padded = history[-max_len:] + [padding_item_id] * max(0, max_len - len(history))
        len_seq = min(len(history), max_len)
        
        session_data.append({
            'user_id': user_id,
            'seq': history_padded,
            'seq_unpad': history[-max_len:],  # 최근 max_len개만 사용
            'len_seq': len_seq,
            'next': next_item_from_history,  # 히스토리의 마지막 아이템 (학습/검증용)
            'next_from_candidates': next_item if gt_ids else None,  # 후보에서의 정답 (테스트용)
            'candidates': candidates,  # 세 번째 컬럼의 후보 (테스트용)
            'has_candidates': len(candidates) > 0  # 후보가 있는지 여부
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Created {len(session_df)} sessions")
    print(f"  - Sessions with candidates (for test): {session_df['has_candidates'].sum()}")
    print(f"  - Sessions without candidates (for train/val): {(~session_df['has_candidates']).sum()}")
    return session_df, padding_item_id

def add_negative_samples(session_df, news_id2name, cans_num=10):
    """각 세션에 negative sampling으로 후보 추가 (학습/검증용)"""
    if len(session_df) == 0:
        print("Warning: Empty DataFrame, skipping negative sampling")
        return session_df
    
    print("Adding negative samples...")
    
    all_item_ids = list(news_id2name.keys())
    
    def sample_candidates(row):
        seq_unpad = row['seq_unpad']
        next_item = row['next']
        
        # seq_unpad가 리스트가 아닌 경우 리스트로 변환
        if not isinstance(seq_unpad, list):
            seq_unpad = list(seq_unpad) if hasattr(seq_unpad, '__iter__') else [seq_unpad]
        
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
    
    # apply 결과를 리스트로 변환하여 할당
    candidates_list = session_df.apply(sample_candidates, axis=1).tolist()
    session_df['candidates'] = candidates_list
    print("Negative sampling complete")
    return session_df

def split_data_by_user(session_df, train_ratio=0.7, val_ratio=0.15):
    """각 사용자별로 train/val/test로 분할
    - 학습/검증: 두 번째 컬럼(히스토리) 데이터 사용 (마지막 아이템이 정답)
    - 테스트: 세 번째 컬럼(후보) 데이터가 있는 세션만 사용
    같은 세션이 학습/검증과 테스트 모두에 사용될 수 있음
    """
    print("Splitting data by user...")
    print("Train/Val: using column 2 (history), Test: using column 3 (candidates)")
    
    # 테스트 데이터: 세 번째 컬럼의 후보가 있는 세션만
    test_df = session_df[session_df['has_candidates'] == True].copy()
    if len(test_df) > 0:
        # 테스트용으로 next를 후보에서의 정답으로 변경
        test_df['next'] = test_df['next_from_candidates']
        # 불필요한 컬럼 제거
        test_df = test_df.drop(columns=['next_from_candidates', 'has_candidates'], errors='ignore')
    
    # 학습/검증 데이터: 두 번째 컬럼의 모든 세션 사용
    # 세 번째 컬럼이 있어도 두 번째 컬럼 데이터로 학습/검증에 사용 가능
    # (같은 세션이 학습/검증과 테스트 모두에 사용될 수 있음)
    train_val_df = session_df.copy()  # 모든 세션 사용
    if len(train_val_df) > 0:
        # candidates 컬럼은 나중에 negative sampling으로 덮어쓸 예정이므로 제거하지 않음
        # 불필요한 컬럼만 제거
        train_val_df = train_val_df.drop(columns=['next_from_candidates', 'has_candidates'], errors='ignore')
    
    # 학습/검증 데이터를 100% 사용하여 train/val로 분할
    # 각 사용자의 히스토리 시퀀스에서 마지막 아이템을 검증용으로, 나머지를 학습용으로
    # 각 사용자는 한 줄(한 세션)만 있으므로, 각 세션의 히스토리를 분할
    train_sessions = []
    val_sessions = []
    
    # 각 행(세션)을 처리
    for idx, row in train_val_df.iterrows():
        user_id = row['user_id']
        seq_unpad = row['seq_unpad']  # 히스토리 시퀀스 (리스트)
        
        if not isinstance(seq_unpad, list):
            seq_unpad = list(seq_unpad) if hasattr(seq_unpad, '__iter__') else [seq_unpad]
        
        if len(seq_unpad) < 2:
            # 히스토리가 너무 짧으면 모두 학습용으로
            train_sessions.append(pd.DataFrame([row]))
            continue
        
        # 히스토리에서 마지막 아이템을 검증용으로, 나머지를 학습용으로
        # 학습용: 초기 아이템들 (마지막 2개 제외) → 마지막에서 두 번째 아이템 예측
        # 검증용: 마지막 아이템 제외한 히스토리 → 마지막 아이템 예측
        
        # 예: [N1, N2, N3, N4, N5]
        # 학습용: [N1, N2, N3] → N4 예측
        # 검증용: [N1, N2, N3, N4] → N5 예측
        
        if len(seq_unpad) < 3:
            # 히스토리가 너무 짧으면 모두 학습용으로
            train_sessions.append(pd.DataFrame([row]))
            continue
        
        # 학습용: 마지막 2개 아이템 제외한 히스토리 → 마지막에서 두 번째 아이템 예측
        train_history = seq_unpad[:-2]
        train_next = seq_unpad[-2]  # 마지막에서 두 번째 아이템이 정답
        
        # 검증용: 마지막 아이템 제외한 히스토리 → 마지막 아이템 예측
        val_history = seq_unpad[:-1]  # 마지막 아이템 제외
        val_next = seq_unpad[-1]  # 마지막 아이템이 정답
        
        # 학습용 세션 생성
        train_row = row.copy()
        train_row['seq_unpad'] = train_history
        train_row['next'] = train_next
        # 패딩 재계산
        max_len = 50
        padding_item_id = 130319  # 기본값, 필요시 row에서 가져올 수 있음
        train_history_padded = train_history[-max_len:] + [padding_item_id] * max(0, max_len - len(train_history))
        train_row['seq'] = train_history_padded
        train_row['len_seq'] = min(len(train_history), max_len)
        train_sessions.append(pd.DataFrame([train_row]))
        
        # 검증용 세션 생성
        val_row = row.copy()
        val_row['seq_unpad'] = val_history
        val_row['next'] = val_next
        # 패딩 재계산
        val_history_padded = val_history[-max_len:] + [padding_item_id] * max(0, max_len - len(val_history))
        val_row['seq'] = val_history_padded
        val_row['len_seq'] = min(len(val_history), max_len)
        val_sessions.append(pd.DataFrame([val_row]))
    
    # 리스트를 DataFrame으로 변환
    if train_sessions:
        train_df = pd.concat(train_sessions, ignore_index=True)
    else:
        train_df = pd.DataFrame()
    
    if val_sessions:
        val_df = pd.concat(val_sessions, ignore_index=True)
    else:
        val_df = pd.DataFrame()
    
    print(f"Train: {len(train_df)} sessions (using 100% of column 2), Val: {len(val_df)} sessions (using 100% of column 2), Test: {len(test_df)} sessions (using column 3)")
    
    # 사용자별 통계
    train_users = train_df['user_id'].nunique() if len(train_df) > 0 else 0
    val_users = val_df['user_id'].nunique() if len(val_df) > 0 else 0
    test_users = test_df['user_id'].nunique() if len(test_df) > 0 else 0
    print(f"Train users: {train_users}, Val users: {val_users}, Test users: {test_users}")
    
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
    
    # 시퀀스 데이터 생성 (모든 세션에 세 번째 컬럼의 후보 저장)
    session_df, padding_item_id = create_sequences('MIND.tsv', news_id2name)
    
    # 데이터 분할 (각 사용자별로 시간 순서에 따라 분할)
    train_df, val_df, test_df = split_data_by_user(session_df, train_ratio=0.7, val_ratio=0.15)
    
    # 학습 데이터에 negative sampling 적용
    print("\nApplying negative sampling to training data...")
    train_df = add_negative_samples(train_df, news_id2name, cans_num=10)
    
    # 검증 데이터에 negative sampling 적용
    print("\nApplying negative sampling to validation data...")
    val_df = add_negative_samples(val_df, news_id2name, cans_num=10)
    
    # 테스트 데이터는 세 번째 컬럼의 주어진 후보 그대로 사용 (candidates 유지)
    print("\nTest data will use candidates from third column (no negative sampling)")
    
    # 저장
    save_dataframes(train_df, val_df, test_df, news_id2name, 'data/ref/mind')
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"{'='*50}")
    print(f"Padding item ID: {padding_item_id}")
    print(f"Train sessions: {len(train_df)} (with negative sampling)")
    print(f"Val sessions: {len(val_df)} (with negative sampling)")
    print(f"Test sessions: {len(test_df)} (using candidates from column 3)")
    print(f"Output directory: data/ref/mind")

if __name__ == '__main__':
    main()

