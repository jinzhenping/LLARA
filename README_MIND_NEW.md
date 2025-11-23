# MIND New 데이터셋 전처리 가이드

## 개요
`mind_new/behaviors_new.tsv`와 `mind_new/news.tsv` 파일을 사용하여 LLaRA 학습을 위한 데이터를 준비합니다.

## 전처리 단계

### 1. 전처리 스크립트 실행

```bash
python preprocess_mind_new.py
```

이 스크립트는 다음 작업을 수행합니다:
- `mind_new/news.tsv`에서 뉴스 ID → 제목 매핑 생성
- `mind_new/behaviors_new.tsv`에서 사용자 시퀀스 데이터 추출
- 각 시퀀스의 마지막 아이템을 정답으로, 나머지를 히스토리로 설정
- Negative sampling으로 10개 후보 생성
- Train/Val/Test 분할 (70%/15%/15%)
- `data/ref/mind_new/` 디렉토리에 결과 저장

### 2. 출력 파일 확인

전처리 후 다음 파일들이 생성됩니다:
```
data/ref/mind_new/
├── train_data.df      # 학습 데이터 (pickle)
├── Val_data.df        # 검증 데이터 (pickle)
├── Test_data.df       # 테스트 데이터 (pickle)
└── id2name.txt        # 뉴스 ID → 제목 매핑
```

### 3. Padding Item ID 확인

전처리 스크립트 실행 후 출력된 `Padding item ID` 값을 확인합니다.
예: `Padding item ID: 130319`

### 4. main.py 업데이트 (필요시)

만약 padding_item_id가 130319가 아니라면, `main.py`의 다음 부분을 수정:

```python
elif 'mind' in args.data_dir:
    args.padding_item_id = 130319  # 전처리에서 출력된 값으로 변경
```

또는 `mind_new`를 별도로 처리:

```python
elif 'mind_new' in args.data_dir:
    args.padding_item_id = [전처리에서 출력된 값]
```

### 5. 학습 실행

```bash
sh train_mind.sh
```

## 주의사항

1. **SASRec.pth 모델**: 
   - 기존에 학습된 SASRec 모델(`SASRec.pth`)이 있어야 합니다.
   - 이 모델은 새로운 데이터셋의 아이템 ID 범위와 호환되어야 합니다.
   - 만약 아이템 ID 범위가 다르다면, SASRec 모델을 새로 학습해야 할 수 있습니다.

2. **데이터 형식**:
   - `behaviors_new.tsv`: `user_id \t news_id1 news_id2 ...` 형식
   - `news.tsv`: `news_id \t category \t subcategory \t title \t abstract \t ...` 형식

3. **메모리**:
   - 대용량 데이터셋의 경우 메모리 사용량이 클 수 있습니다.
   - 필요시 배치 크기를 줄이거나 데이터 샘플링을 고려하세요.

## 문제 해결

### 전처리 오류
- 파일 경로 확인: `mind_new/behaviors_new.tsv`, `mind_new/news.tsv`가 존재하는지 확인
- 인코딩 문제: 파일이 UTF-8로 인코딩되어 있는지 확인

### 학습 오류
- SASRec 모델 호환성: 아이템 ID 범위가 모델과 일치하는지 확인
- Padding item ID: 전처리에서 출력된 값과 main.py의 값이 일치하는지 확인

