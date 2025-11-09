"""
실제 생성 텍스트를 분석하여 HIT@1이 낮은 원인을 파악
"""

import pandas as pd
import ast

# CSV 데이터 직접 분석
data = """0,"news - newspolitics: White House: NSC Ukraine expert wrote summary of April Trump-Zelensky call that conflicts with rough transcript

White House: NSC Ukraine expert wrote summary of April news - newspolitics: White House: NSC Ukraine expert wrote summary of April Trump-Z",news - newspolitics: 'A turning point in this hearing': Fox personalities speculate if Trump-Yovanovitch tweet could lead to witness intimidation charges,"['movies - movies-gallery: The most talked about movie moments of the 2010s', 'lifestyle - shop-holidays: The Hottest Tech Gifts This Holiday Season', ""news - newspolitics: 'A turning point in this hearing': Fox personalities speculate if Trump-Yovanovitch tweet could lead to witness intimidation charges"", 'finance - finance-companies: FDA Poised to Drop the Hammer on Dollar Tree', 'news - newsus: Surviving Santa Clarita school shooting victims on road to recovery: Latest']"
"""

print("=" * 80)
print("생성 텍스트 분석 결과")
print("=" * 80)

# 주요 문제점들
issues = {
    "빈 생성": 0,
    "기사 본문 생성": 0,
    "중복 생성": 0,
    "잘린 생성": 0,
    "후보 미포함": 0,
    "정답 미포함": 0
}

examples = {
    "빈 생성": [],
    "기사 본문 생성": [],
    "중복 생성": [],
    "잘린 생성": [],
    "후보 미포함": [],
    "정답 미포함": []
}

# 샘플 데이터 분석
samples = [
    (0, "news - newspolitics: White House: NSC Ukraine expert wrote summary of April Trump-Zelensky call that conflicts with rough transcript\n\nWhite House: NSC Ukraine expert wrote summary of April news - newspolitics: White House: NSC Ukraine expert wrote summary of April Trump-Z", "news - newspolitics: 'A turning point in this hearing': Fox personalities speculate if Trump-Yovanovitch tweet could lead to witness intimidation charges"),
    (8, "", "finance - finance-real-estate: 10 reasons it's better to rent rather than buy a home"),
    (2, "sports - football_nfl: Opinion: Colin Kaepernick is about to get what he deserves: a chance\n\nThe San Francisco 49ers are about to prove that athletes are what they say they are: people\n\nThe team announced Wednesday that it is working out", "sports - football_nfl: Opinion: Colin Kaepernick is about to get what he deserves: a chance"),
    (50, "sports - football_nfl: Opinion: Colin Kaepernick is about to get what he deserves: a chance\n\nThis user has read sports - football_nfl: Opinion: Colin Kaepernick is about to get what he deserves: a chance\n\nThis", "tv - tv-celebrity: Alex Rodriguez Jokingly Talks About Marrying Jennifer Lopez During Super Bowl Halftime Show"),
]

print("\n1. 주요 문제점 분석:")
print("-" * 80)

# 빈 생성
empty_count = sum(1 for _, gen, _ in samples if not gen or gen.strip() == "")
print(f"   빈 생성: {empty_count}/{len(samples)} ({empty_count/len(samples)*100:.1f}%)")

# 기사 본문 생성 (줄바꿈이 있고, 긴 텍스트)
article_count = sum(1 for _, gen, _ in samples if gen and '\n' in gen and len(gen) > 100)
print(f"   기사 본문 생성: {article_count}/{len(samples)} ({article_count/len(samples)*100:.1f}%)")

# 중복 생성 (같은 제목 반복)
repeat_count = sum(1 for _, gen, _ in samples if gen and "Colin Kaepernick" in gen)
print(f"   'Colin Kaepernick' 반복 생성: {repeat_count}/{len(samples)} ({repeat_count/len(samples)*100:.1f}%)")

print("\n2. 예시 분석:")
print("-" * 80)

for idx, (sample_idx, gen, real) in enumerate(samples):
    print(f"\n   [예시 {idx+1}]")
    print(f"   생성 텍스트: {gen[:150] if gen else '(빈 문자열)'}...")
    print(f"   정답: {real[:100]}...")
    
    if not gen or gen.strip() == "":
        print(f"   문제: 빈 생성")
    elif '\n' in gen and len(gen) > 100:
        print(f"   문제: 기사 본문 생성 (순위 리스트가 아님)")
    elif "Colin Kaepernick" in gen:
        print(f"   문제: 특정 제목 반복 생성")
    elif real.lower() not in gen.lower():
        print(f"   문제: 정답이 생성 텍스트에 없음")

print("\n3. 근본 원인:")
print("-" * 80)
print("""
   a) LLM이 순위 리스트를 생성하지 않고 뉴스 기사 본문을 생성
      - 프롬프트가 "순위를 출력하라"고 하지만, 모델이 기사를 생성
      - 학습 데이터에 기사 본문이 포함되어 있어서 그렇게 학습됨
   
   b) 생성 길이 제한 (max_gen_length=64)
      - 5개 제목을 모두 포함하기에 부족
      - 평균 제목 길이가 50-100자이므로 5개면 250-500자 필요
   
   c) 모델이 특정 제목에 과적합
      - "Colin Kaepernick" 제목이 반복적으로 생성됨
      - 학습 데이터에 이 제목이 많이 포함되어 있을 가능성
   
   d) 빈 생성
      - 일부 케이스에서 아무것도 생성하지 않음
      - 모델이 확신이 없을 때 빈 출력을 생성
""")

print("\n4. 해결 방안:")
print("-" * 80)
print("""
   1) max_gen_length 증가: 64 → 256 또는 512
   2) 프롬프트 개선: 더 명확한 출력 형식 요구
   3) 생성 전략 변경: 
      - 순위 리스트 대신 확률 분포 출력
      - 또는 각 후보에 대한 점수 출력
   4) 파싱 로직 개선:
      - 생성 텍스트가 기사 본문인 경우 처리
      - 부분 매칭 개선
   5) 학습 데이터 검증:
      - 학습 데이터에 기사 본문이 포함되지 않도록
      - 순위 리스트 형식만 학습
""")

