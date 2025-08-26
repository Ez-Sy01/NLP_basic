# 📝 Text Classification Overview

텍스트 분류(Text Classification)는 문서나 문장을 특정 카테고리로 자동 분류하는 기술입니다.  
스팸 메일 필터링, 감정 분석, 뉴스 카테고리 분류 등 다양한 분야에서 활용됩니다.  

아래는 텍스트 분류 방법의 발전 흐름을 정리한 내용입니다.  

---

## 1. 전통적 기법 (Classical ML)
- **Bag-of-Words (BoW) + ML 분류기**
  - 단순 단어 등장 횟수 기반 → Logistic Regression, SVM
  - 장점: 빠르고 단순
  - 단점: 문맥 고려 불가
  - 예시: `CountVectorizer + Logistic Regression`

- **TF-IDF + Naive Bayes / SVM**
  - 흔한 단어는 낮은 가중치, 희귀 단어는 높은 가중치
  - 스팸 메일 필터, 뉴스 분류에서 자주 활용

---

## 2. 단어 임베딩 기반 (Word Embedding)
- **Word2Vec / GloVe**
  - 단어를 벡터로 표현 → 유사 단어끼리 가까움
  - 예: `king - man + woman ≈ queen`
  - 분류 시 평균 벡터 → Dense Layer 활용

- **FastText (Facebook)**
  - Word2Vec 확장판 → subword 기반
  - 한국어 등 형태소 기반 언어에서 성능 우수

---

## 3. 딥러닝 기반
- **CNN for Text (Kim CNN, 2014)**
  - 문장을 단어 임베딩 시퀀스로 보고 CNN 적용
  - 키워드/패턴(n-gram feature)에 강함

- **RNN (LSTM/GRU)**
  - 순서와 문맥 반영 가능
  - 긴 문장 처리 가능하지만 gradient 문제로 한계 존재

---

## 4. Transformer 기반 (최신)
- **BERT (2018)**
  - 양방향 문맥 이해 가능
  - Fine-tuning → 문장 분류, 감정 분석, QA 등 대부분 가능
  - 한국어 모델: KoBERT, KR-BERT, KLUE-BERT

- **DistilBERT, ALBERT**
  - 경량화 모델 → 모바일/실시간 환경 적합

- **RoBERTa, ELECTRA**
  - 학습 방식을 개선해 성능 향상

---

## 5. 최신 응용
- **Prompt 기반 분류 (LLM 활용)**
  - ChatGPT, GPT-4 같은 대형 언어모델 사용
  - Zero-shot classification 가능  
    예:  
    ```python
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification")
    classifier("I love this movie!", candidate_labels=["positive", "negative"])
    ```
  - 별도 학습 없이도 분류 가능

