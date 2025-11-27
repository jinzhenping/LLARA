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
import argparse
import argparse

def load_news_mapping(news_file='mind_new/news.tsv', behaviors_file=None):
    """뉴스 ID를 뉴스 제목으로 매핑하는 딕셔너리 생성
    behaviors_file에서 실제로 사용되는 뉴스 ID만 매핑에 포함
    """
    print("Loading news mapping...")
    
    # 먼저 behaviors 파일에서 사용되는 모든 뉴스 ID 수집
    used_news_ids = set()
    if behaviors_file:
        print(f"Scanning {behaviors_file} for used news IDs...")
        try:
            with open(behaviors_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line_idx % 10000 == 0 and line_idx > 0:
                        print(f"  Scanning line {line_idx}...")
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        seq_str = parts[1].strip()  # 두 번째 컬럼
                        for nid_str in seq_str.split():
                            nid = parse_news_id(nid_str.strip())
                            if nid is not None:
                                used_news_ids.add(nid)
        except Exception as e:
            print(f"Warning: Could not scan behaviors file: {e}")
    
    # news.tsv에서 매핑 로드 (사용되는 ID만)
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
    
    print(f"Loaded {len(news_id2name)} news items from news.tsv")
    if behaviors_file and used_news_ids:
        matched = len(set(news_id2name.keys()) & used_news_ids)
        print(f"Found {len(used_news_ids)} unique news IDs in behaviors file")
        print(f"Matched {matched} news items that are used in behaviors file")
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

def create_sequences_from_behaviors_new(behaviors_file='mind_new/behaviors_new.tsv', 
                                        news_id2name=None, 
                                        padding_item_id=None, 
                                        min_seq_len=3, 
                                        max_seq_len=50,
                                        num_users=None):
    """behaviors_new.tsv를 읽어서 시퀀스 데이터 생성 (첫 번째, 두 번째 컬럼만 사용)
    - 첫 번째 컬럼: 사용자 ID
    - 두 번째 컬럼: 히스토리 시퀀스 (마지막 아이템이 정답)
    - num_users: 사용할 사용자 수 (None이면 전체)
    """
    print(f"Loading {behaviors_file}...")
    print("Using columns: 1=user_id, 2=history sequence (last item is ground truth)")
    if num_users:
        print(f"Using first {num_users} users")
    
    # 모든 뉴스 ID 수집하여 padding_item_id 결정
    all_news_ids = set()
    session_data_raw = []
    unique_users = set()
    
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 10000 == 0 and line_idx > 0:
                print(f"Processing line {line_idx}...")
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            user_id = parts[0].strip()
            
            # num_users 제한이 있고 이미 충분한 사용자를 수집한 경우
            if num_users and len(unique_users) >= num_users:
                if user_id not in unique_users:
                    continue  # 새로운 사용자는 스킵
            
            unique_users.add(user_id)
            
            # num_users 제한이 있고 아직 충분하지 않은 경우
            if num_users and len(unique_users) > num_users:
                break  # 충분한 사용자를 수집했으면 중단
            
            seq_str = parts[1].strip()  # 두 번째 컬럼: 공백으로 구분된 뉴스 ID 리스트
            
            # 시퀀스 파싱
            seq_ids = []
            for nid_str in seq_str.split():
                nid = parse_news_id(nid_str.strip())
                if nid is not None:
                    seq_ids.append(nid)
                    all_news_ids.add(nid)
            
            if len(seq_ids) < min_seq_len:
                continue
            
            # 마지막 아이템을 정답으로, 나머지를 히스토리로
            history = seq_ids[:-1]
            next_item = seq_ids[-1]
            
            session_data_raw.append({
                'user_id': user_id,
                'seq_ids': history,
                'next_item': next_item
            })
    
    print(f"Found {len(unique_users)} unique users")
    print(f"Total unique news items: {len(all_news_ids)}")
    print(f"Total sessions: {len(session_data_raw)}")
    
    # padding_item_id는 이미 main에서 설정되었으므로 여기서는 사용하지 않음
    
    # 시퀀스 데이터 생성
    session_data = []
    
    for session in session_data_raw:
        seq_ids = session['seq_ids']
        next_item = session['next_item']
        
        if len(seq_ids) < min_seq_len - 1:  # 최소 히스토리 길이 (정답 제외)
            continue
        
        # 모든 아이템이 news_id2name에 있는지 확인
        if next_item not in news_id2name:
            continue
        if not all(nid in news_id2name for nid in seq_ids):
            continue
        
        # 패딩 추가 (최대 길이 제한)
        history_padded = seq_ids[-max_seq_len:] + [padding_item_id] * max(0, max_seq_len - len(seq_ids))
        len_seq = min(len(seq_ids), max_seq_len)
        
        session_data.append({
            'user_id': session['user_id'],
            'seq': history_padded,
            'seq_unpad': seq_ids[-max_seq_len:],  # 최근 max_seq_len개만 사용
            'len_seq': len_seq,
            'next': next_item,
            'candidates': []  # 나중에 negative sampling으로 채워짐
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Created {len(session_df)} sessions from behaviors_new.tsv")
    return session_df

def create_sequences(behaviors_file='mind_new/behaviors_194_users.tsv', news_id2name=None, padding_item_id=None, min_seq_len=3, max_seq_len=50):
    """behaviors_194_users.tsv를 읽어서 시퀀스 데이터 생성
    - 첫 번째 컬럼: 사용자 ID
    - 두 번째 컬럼: 사용자 히스토리
    - 세 번째 컬럼: 5개 후보 (첫 번째가 정답)
    """
    print(f"Loading {behaviors_file}...")
    print("Using columns: 1=user_id, 2=history, 3=candidates (first candidate is ground truth)")
    
    # 모든 뉴스 ID 수집하여 padding_item_id 결정
    all_news_ids = set()
    session_data_raw = []
    
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 1000 == 0 and line_idx > 0:
                print(f"Processing line {line_idx}...")
            
            parts = line.strip().split('\t')
            if len(parts) < 3:
                # 세 번째 컬럼이 없는 경우 스킵
                continue
            
            user_id = parts[0].strip()
            seq_str = parts[1].strip()  # 두 번째 컬럼: 공백으로 구분된 뉴스 ID 리스트 (히스토리)
            candidates_str = parts[2].strip()  # 세 번째 컬럼: 공백으로 구분된 5개 후보 (첫 번째가 정답)
            
            # 히스토리 시퀀스 파싱
            seq_ids = []
            for nid_str in seq_str.split():
                nid = parse_news_id(nid_str.strip())
                if nid is not None:
                    seq_ids.append(nid)
                    all_news_ids.add(nid)
            
            # 후보 파싱 (세 번째 컬럼)
            candidate_ids = []
            for nid_str in candidates_str.split():
                nid = parse_news_id(nid_str.strip())
                if nid is not None:
                    candidate_ids.append(nid)
                    all_news_ids.add(nid)
            
            if len(seq_ids) < min_seq_len:
                continue
            
            if len(candidate_ids) == 0:
                # 후보가 없는 경우 스킵
                continue
            
            # 첫 번째 후보가 정답
            next_item = candidate_ids[0]
            candidates = candidate_ids  # 5개 후보 전체
            
            session_data_raw.append({
                'user_id': user_id,
                'seq_ids': seq_ids,
                'next_item': next_item,
                'candidates': candidates
            })
    
    if padding_item_id is None:
        padding_item_id = max(all_news_ids) + 1 if all_news_ids else 130319
    
    print(f"Padding item ID: {padding_item_id}")
    print(f"Total unique news items: {len(all_news_ids)}")
    print(f"Total sessions: {len(session_data_raw)}")
    
    # 시퀀스 데이터 생성
    session_data = []
    
    for session in session_data_raw:
        seq_ids = session['seq_ids']
        next_item = session['next_item']
        candidates = session['candidates']
        
        if len(seq_ids) < min_seq_len:
            continue
        
        # 히스토리는 두 번째 컬럼 전체 사용
        history = seq_ids
        
        # 모든 아이템이 news_id2name에 있는지 확인
        if next_item not in news_id2name:
            continue
        if not all(nid in news_id2name for nid in history):
            continue
        if not all(cid in news_id2name for cid in candidates):
            continue
        
        # 패딩 추가 (최대 길이 제한)
        history_padded = history[-max_seq_len:] + [padding_item_id] * max(0, max_seq_len - len(history))
        len_seq = min(len(history), max_seq_len)
        
        session_data.append({
            'user_id': session['user_id'],
            'seq': history_padded,
            'seq_unpad': history[-max_seq_len:],  # 최근 max_seq_len개만 사용
            'len_seq': len_seq,
            'next': next_item,
            'candidates': candidates  # 세 번째 컬럼의 5개 후보 (첫 번째가 정답)
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Created {len(session_df)} sessions")
    return session_df, padding_item_id, all_news_ids

def add_negative_samples(session_df, news_id2name, cans_num=10, use_existing_candidates=True):
    """각 세션에 negative sampling으로 후보 추가
    use_existing_candidates=True: 이미 candidates가 있으면 그대로 사용 (테스트/검증용)
    use_existing_candidates=False: negative sampling 수행 (학습용)
    """
    # session_df가 tuple인 경우 첫 번째 요소만 사용
    if isinstance(session_df, tuple):
        session_df = session_df[0]
        print("Warning: session_df was a tuple, using first element")
    
    if use_existing_candidates:
        # 이미 candidates가 있는 경우 (세 번째 컬럼에서 가져온 경우)
        # 학습 시에만 negative sampling 수행
        print("Candidates already exist from third column. Will use for test/val, negative sampling for train.")
        return session_df
    
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
    parser = argparse.ArgumentParser(description='Preprocess MIND dataset')
    parser.add_argument('--num_users', type=int, default=None,
                        help='Number of users to add from behaviors_new.tsv to training data (default: None, use all)')
    args = parser.parse_args()
    
    # 파일 경로 설정
    behaviors_file = 'mind_new/behaviors_194_users.tsv'  # 컬럼 1: user_id, 컬럼 2: history, 컬럼 3: 5 candidates (첫 번째가 정답)
    behaviors_new_file = 'mind_new/behaviors_new.tsv'  # 추가 학습 데이터
    news_file = 'mind_new/news.tsv'
    output_dir = 'data/ref/mind_194_users'
    
    # 뉴스 매핑 로드
    print("="*50)
    print("Step 1: Loading news mapping...")
    print("="*50)
    news_id2name, news_id2idx = load_news_mapping(news_file, behaviors_file)
    
    if len(news_id2name) == 0:
        print("Error: No news items loaded. Check the news.tsv file format.")
        return
    
    # behaviors_194_users.tsv에서 테스트/검증 데이터 생성
    print("\n" + "="*50)
    print("Step 2: Loading behaviors_194_users.tsv (for test/val)...")
    print("="*50)
    session_df, padding_item_id, all_news_ids = create_sequences(
        behaviors_file, 
        news_id2name,
        padding_item_id=None,
        min_seq_len=3,
        max_seq_len=50
    )
    
    if len(session_df) == 0:
        print("Error: No sessions created. Check the behaviors_194_users.tsv file format.")
        return
    
    # behaviors_194_users.tsv의 모든 데이터를 테스트 데이터로 사용
    # 전체를 test로만 사용 (val은 일부만, train은 behaviors_new.tsv에서만)
    print("\n" + "="*50)
    print("Step 2.5: Using all behaviors_194_users.tsv data for test...")
    print("="*50)
    # 전체를 test로 사용하고, val은 일부만 (또는 test만 사용)
    # 사용자 요청: 전체 194개를 테스트로 사용
    test_df = session_df.copy()  # 전체를 테스트로
    val_df = session_df.sample(frac=0.15, random_state=42).copy()  # 검증용으로 일부만 (선택사항)
    train_df_194 = pd.DataFrame()  # 빈 DataFrame (behaviors_194_users.tsv에서는 학습 데이터 없음)
    print(f"Test: {len(test_df)} (all data), Val: {len(val_df)} (subset for validation)")
    
    # behaviors_new.tsv에서 추가 학습 데이터 생성
    if args.num_users is not None and args.num_users > 0:
        print("\n" + "="*50)
        print(f"Step 3: Loading behaviors_new.tsv (adding {args.num_users} users to training data)...")
        print("="*50)
        train_df_new = create_sequences_from_behaviors_new(
            behaviors_new_file,
            news_id2name,
            padding_item_id=padding_item_id,  # 기존 padding_item_id 사용
            min_seq_len=3,
            max_seq_len=50,
            num_users=args.num_users
        )
        
        # train_df_new가 DataFrame인지 확인
        if not isinstance(train_df_new, pd.DataFrame):
            print(f"Error: train_df_new is not a DataFrame. Type: {type(train_df_new)}")
            if isinstance(train_df_new, tuple):
                print(f"  Tuple length: {len(train_df_new)}")
                train_df_new = train_df_new[0]  # 첫 번째 요소만 사용
            else:
                return
        
        # Negative sampling 적용
        print("\nApplying negative sampling to additional training data...")
        train_df_new = add_negative_samples(train_df_new, news_id2name, cans_num=10, use_existing_candidates=False)
        
        # behaviors_194_users.tsv에서는 학습 데이터가 없으므로 train_df_new만 사용
        print(f"\nTraining data:")
        print(f"  - From behaviors_new.tsv ({args.num_users} users): {len(train_df_new)} sessions")
        train_df = train_df_new
        print(f"  - Total: {len(train_df)} sessions")
    else:
        print("\n" + "="*50)
        print("Step 3: Skipping behaviors_new.tsv (num_users not specified)")
        print("="*50)
        print("WARNING: No training data! Please specify --num_users to add training data from behaviors_new.tsv")
        # 빈 학습 데이터 (경고)
        train_df = pd.DataFrame()
        print(f"Training data: {len(train_df)} sessions (empty!)")
    
    # 검증/테스트 데이터는 세 번째 컬럼의 후보 그대로 사용 (이미 candidates에 저장됨)
    print("\nValidation and test data will use candidates from third column (first candidate is ground truth)")
    
    # 저장
    print("\n" + "="*50)
    print("Step 4: Saving data...")
    print("="*50)
    save_dataframes(train_df, val_df, test_df, news_id2name, output_dir)
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"{'='*50}")
    print(f"Test/Val input file: {behaviors_file}")
    print(f"  - Column 1: user_id")
    print(f"  - Column 2: history sequence")
    print(f"  - Column 3: 5 candidates (first is ground truth)")
    if args.num_users:
        print(f"\nAdditional training data: {behaviors_new_file}")
        print(f"  - Added {args.num_users} users")
        print(f"  - Column 1: user_id")
        print(f"  - Column 2: history sequence (last item is ground truth)")
    print(f"\nPadding item ID: {padding_item_id}")
    print(f"Total news items: {len(news_id2name)}")
    print(f"Train sessions: {len(train_df)} (with negative sampling)")
    print(f"Val sessions: {len(val_df)} (using candidates from column 3)")
    print(f"Test sessions: {len(test_df)} (using candidates from column 3)")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Update train_mind.sh to use data_dir='{output_dir}'")
    print(f"2. Update padding_item_id in main.py if needed (current: {padding_item_id})")
    print(f"3. Run: sh train_mind.sh")

if __name__ == '__main__':
    main()
