# 📝 Text Classification & Preprocessing Overview

텍스트 분류(Text Classification)는 문서나 문장을 특정 카테고리로 자동 분류하는 기술입니다.  
스팸 메일 필터링, 감정 분석, 뉴스 카테고리 분류 등 다양한 분야에서 활용됩니다.  

아래는 텍스트 분류 방법과 전처리 과정, 최신 NLP 모델 활용법을 한눈에 정리한 문서입니다.  

---

## 1. 텍스트 분류 방법

### 1.1 전통적 기법 (Classical ML)

- **Bag-of-Words (BoW) + ML 분류기**  
  - 단순 단어 등장 횟수 기반 → Logistic Regression, SVM  
  - 장점: 빠르고 단순  
  - 단점: 문맥 고려 불가  
  - 예시: `CountVectorizer + Logistic Regression`

- **TF-IDF + Naive Bayes / SVM**  
  - 흔한 단어는 낮은 가중치, 희귀 단어는 높은 가중치  
  - 스팸 메일 필터, 뉴스 분류에서 자주 활용

---

### 1.2 단어 임베딩 기반 (Word Embedding)

- **Word2Vec / GloVe**  
  - 단어를 벡터로 표현 → 유사 단어끼리 가까움  
  - 예: `king - man + woman ≈ queen`  
  - 분류 시 평균 벡터 → Dense Layer 활용

- **FastText (Facebook)**  
  - Word2Vec 확장판 → subword 기반  
  - 한국어 등 형태소 기반 언어에서 성능 우수

---

### 1.3 딥러닝 기반

- **CNN for Text (Kim CNN, 2014)**  
  - 문장을 단어 임베딩 시퀀스로 보고 CNN 적용  
  - 키워드/패턴(n-gram feature)에 강함

- **RNN (LSTM/GRU)**  
  - 순서와 문맥 반영 가능  
  - 긴 문장 처리 가능하지만 gradient 문제로 한계 존재

---

### 1.4 Transformer 기반 (최신)

- **BERT (2018)**  
  - 양방향 문맥 이해 가능  
  - Fine-tuning → 문장 분류, 감정 분석, QA 등 대부분 가능  
  - 한국어 모델: KoBERT, KR-BERT, KLUE-BERT

- **DistilBERT, ALBERT**  
  - 경량화 모델 → 모바일/실시간 환경 적합

- **RoBERTa, ELECTRA**  
  - 학습 방식을 개선해 성능 향상

---
### 1.5 최신 응용 - **Prompt 기반 분류 (LLM 활용)** - ChatGPT, GPT-4 같은 대형 언어모델 사용 - Zero-shot classification 가능
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "I love this movie!",
    candidate_labels=["positive", "negative"]
)
```
---
# 🔧 Text Preprocessing for NLP

텍스트 분류(Text Classification) 모델을 학습하기 전에 반드시 거쳐야 하는 과정이 **텍스트 전처리(Text Preprocessing)**입니다.  
언어 데이터의 특성상 불필요한 기호, 중복 표현, 대소문자 문제 등이 존재하기 때문에, 모델 성능을 높이기 위해 데이터를 정제하는 것이 매우 중요합니다.  

---

## 1. 기본 정제 (Cleaning)

- 불필요한 기호, 특수문자 제거  
  - 예: `, . ! ?` 같은 문장부호 제거
- 숫자 처리  
  - 그대로 둘지 / 토큰 `<NUM>`으로 대체할지 결정
- 대소문자 통일  
  - 영어의 경우 대부분 소문자 변환 (`Apple` → `apple`)

---

## 2. 토큰화 (Tokenization)

문장을 단어 또는 의미 단위로 나누는 과정입니다.

- **공백 기반 토큰화**  
  - 단순하지만 한국어에선 한계 존재
- **형태소 분석기 활용 (한국어)**  
  - Mecab, Konlpy, Okt 등
  - 예: `나는 학교에 간다` → `['나', '는', '학교', '에', '가', 'ᆫ다']`
- **Subword 토큰화 (WordPiece, BPE)**  
  - 신조어, 희귀 단어 처리에 강함
  - BERT 계열 모델에서 활용됨

---

## 3. 불용어 제거 (Stopword Removal)

분류에 큰 의미가 없는 단어 제거  

- 한국어: `은`, `는`, `이`, `가`, `그리고`, `하지만`  
- 영어: `is`, `the`, `and`, `but`  

---

## 4. 정규화 (Normalization)

- **어간 추출 (Stemming)**  
  - 단어의 원형으로 변환 (예: `running` → `run`)  
  - 단순 규칙 기반이라 부정확할 수 있음
- **표제어 추출 (Lemmatization)**  
  - 품사를 고려한 원형 변환 (`better` → `good`)  
- 한국어의 경우: 활용형 → 기본형 변환 (`갔다` → `가다`)  

---

## 5. 인코딩 (Encoding)

텍스트를 숫자로 변환하는 과정  

- **BoW (Bag of Words)**  
  - 단어 등장 횟수 기반 벡터화
- **TF-IDF**  
  - 중요한 단어에 높은 가중치 부여
- **Word2Vec / FastText / GloVe**  
  - 단어 임베딩 활용
- **Subword 기반 임베딩 (BERT 등)**  
  - Transformer 계열에서 주로 사용

---

## 6. Hugging Face AutoTokenizer

`AutoTokenizer`는 Hugging Face Transformers 라이브러리에서 제공하는 **자동 토크나이저**입니다.  
모델 이름만 지정하면 해당 모델에 맞는 토크나이저를 자동으로 불러와, 토큰화(tokenization), 패딩(padding), 인덱스 변환 등 NLP 모델 입력 전처리를 손쉽게 수행할 수 있습니다.

```python
from transformers import AutoTokenizer

# BERT 모델용 토크나이저 자동 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Natural Language Processing!"
tokens = tokenizer(text)
print(tokens)
# {'input_ids': [...], 'token_type_ids': [...], 'attention_mask': [...]}

7. 전처리 예시 코드 (Python)
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 기본 정제
text = "I love Natural Language Processing! :) 2025"
text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
print(text)  # i love natural language processing

# 토큰화
tokens = word_tokenize(text)
print(tokens)  # ['i', 'love', 'natural', 'language', 'processing']

# 불용어 제거
stop_words = set(stopwords.words("english"))
tokens = [w for w in tokens if w not in stop_words]
print(tokens)  # ['love', 'natural', 'language', 'processing']

✅ 정리

텍스트 전처리는 다음 단계로 구성됩니다:

기본 정제 → 특수문자, 숫자 처리, 대소문자 변환

토큰화 → 단어/형태소/서브워드 단위 분할

불용어 제거 → 의미 없는 단어 제거

정규화 → 어간 추출, 표제어 추출

인코딩 → BoW, TF-IDF, Word2Vec, BERT 토큰화

AutoTokenizer → Hugging Face 모델용 자동 토큰화, 패딩, 텐서 변환 지원
