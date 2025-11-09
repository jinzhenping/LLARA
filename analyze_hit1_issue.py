"""
HIT@1이 낮은 원인을 분석하는 스크립트
생성된 텍스트와 파싱 결과를 확인하여 문제점을 찾습니다.
"""

import pandas as pd
import os.path as op
import json

def analyze_generation_quality(csv_path):
    """생성 품질 분석"""
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("HIT@1 낮은 원인 분석")
    print("=" * 80)
    
    # 1. 생성 텍스트 길이 분석
    df['generate_len'] = df['generate'].str.len()
    print(f"\n1. 생성 텍스트 길이 통계:")
    print(f"   평균: {df['generate_len'].mean():.1f} 문자")
    print(f"   최소: {df['generate_len'].min()} 문자")
    print(f"   최대: {df['generate_len'].max()} 문자")
    print(f"   64자 미만: {(df['generate_len'] < 64).sum()} / {len(df)} ({(df['generate_len'] < 64).sum()/len(df)*100:.1f}%)")
    
    # 2. 후보 개수 확인
    df['num_candidates'] = df['cans'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
    print(f"\n2. 후보 개수:")
    print(f"   평균: {df['num_candidates'].mean():.1f}개")
    print(f"   최소: {df['num_candidates'].min()}개")
    print(f"   최대: {df['num_candidates'].max()}개")
    
    # 3. 정답이 생성 텍스트에 포함되는지 확인
    def check_answer_in_generate(row):
        generate_lower = str(row['generate']).lower()
        real_lower = str(row['real']).lower().strip()
        return real_lower in generate_lower
    
    df['answer_in_generate'] = df.apply(check_answer_in_generate, axis=1)
    answer_found = df['answer_in_generate'].sum()
    print(f"\n3. 정답이 생성 텍스트에 포함되는지:")
    print(f"   포함됨: {answer_found} / {len(df)} ({answer_found/len(df)*100:.1f}%)")
    print(f"   포함 안됨: {len(df) - answer_found} / {len(df)} ({(len(df)-answer_found)/len(df)*100:.1f}%)")
    
    # 4. 생성 텍스트에 포함된 후보 개수 확인
    def count_candidates_in_generate(row):
        generate_lower = str(row['generate']).lower()
        cans = eval(row['cans']) if isinstance(row['cans'], str) else row['cans']
        cans_lower = [str(c).lower().strip() for c in cans]
        found = sum(1 for c in cans_lower if c in generate_lower)
        return found, len(cans)
    
    df[['candidates_found', 'total_candidates']] = df.apply(
        lambda row: pd.Series(count_candidates_in_generate(row)), axis=1
    )
    print(f"\n4. 생성 텍스트에 포함된 후보 개수:")
    print(f"   평균: {df['candidates_found'].mean():.1f} / {df['total_candidates'].mean():.1f}개")
    print(f"   모든 후보 포함: {(df['candidates_found'] == df['total_candidates']).sum()} / {len(df)} ({(df['candidates_found'] == df['total_candidates']).sum()/len(df)*100:.1f}%)")
    
    # 5. 생성 텍스트가 너무 짧은 경우 확인
    short_generations = df[df['generate_len'] < 50]
    if len(short_generations) > 0:
        print(f"\n5. 생성 텍스트가 50자 미만인 경우: {len(short_generations)}개")
        print("   예시:")
        for idx, row in short_generations.head(3).iterrows():
            print(f"   - 생성: {row['generate'][:100]}")
            print(f"     정답: {row['real']}")
    
    # 6. 정답이 생성 텍스트에 없는 경우 상세 분석
    missing_answer = df[~df['answer_in_generate']]
    if len(missing_answer) > 0:
        print(f"\n6. 정답이 생성 텍스트에 없는 경우: {len(missing_answer)}개")
        print("   예시:")
        for idx, row in missing_answer.head(3).iterrows():
            print(f"   - 생성: {row['generate'][:200]}")
            print(f"     정답: {row['real']}")
            cans = eval(row['cans']) if isinstance(row['cans'], str) else row['cans']
            print(f"     후보: {cans[:3]}...")
    
    # 7. 생성 텍스트 형식 분석
    def check_format(row):
        generate = str(row['generate']).lower()
        lines = [l.strip() for l in generate.split('\n') if l.strip()]
        return len(lines), generate.count('\n')
    
    df[['num_lines', 'newline_count']] = df.apply(
        lambda row: pd.Series(check_format(row)), axis=1
    )
    print(f"\n7. 생성 텍스트 형식:")
    print(f"   평균 줄 수: {df['num_lines'].mean():.1f}")
    print(f"   줄바꿈 없는 경우: {(df['num_lines'] == 1).sum()} / {len(df)} ({(df['num_lines'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"   여러 줄: {(df['num_lines'] > 1).sum()} / {len(df)} ({(df['num_lines'] > 1).sum()/len(df)*100:.1f}%)")
    
    # 8. 후보 제목 길이 확인
    def get_avg_title_length(row):
        cans = eval(row['cans']) if isinstance(row['cans'], str) else row['cans']
        if isinstance(cans[0], str):
            return sum(len(str(c)) for c in cans) / len(cans)
        return 0
    
    df['avg_title_length'] = df.apply(get_avg_title_length, axis=1)
    print(f"\n8. 후보 제목 평균 길이:")
    print(f"   평균: {df['avg_title_length'].mean():.1f} 문자")
    print(f"   최대: {df['avg_title_length'].max():.1f} 문자")
    
    # 9. 예시 출력 (정답이 첫 번째에 있는 경우와 없는 경우)
    print(f"\n9. 예시 케이스:")
    print("\n   [케이스 1: 정답이 생성 텍스트에 포함된 경우]")
    found_examples = df[df['answer_in_generate']].head(2)
    for idx, row in found_examples.iterrows():
        print(f"   생성: {row['generate'][:150]}")
        print(f"   정답: {row['real']}")
        print()
    
    print("\n   [케이스 2: 정답이 생성 텍스트에 없는 경우]")
    not_found_examples = df[~df['answer_in_generate']].head(2)
    for idx, row in not_found_examples.iterrows():
        print(f"   생성: {row['generate'][:150]}")
        print(f"   정답: {row['real']}")
        cans = eval(row['cans']) if isinstance(row['cans'], str) else row['cans']
        print(f"   후보: {', '.join(cans[:3])}...")
        print()
    
    return df

if __name__ == '__main__':
    # test.csv 파일 경로
    csv_path = './output/mind_ranking/test.csv'
    
    if not op.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        print("먼저 테스트를 실행하여 test.csv 파일을 생성하세요.")
    else:
        df = analyze_generation_quality(csv_path)
        print("\n" + "=" * 80)
        print("분석 완료!")
        print("=" * 80)

