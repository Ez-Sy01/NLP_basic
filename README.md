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

# 명시적으로 모델과 revision 지정
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    revision="d7645e1"
)

result = classifier(
    "I love this movie!",
    candidate_labels=["positive", "negative"]
)

print(result)
```

---
# 🔧 Text Preprocessing for NLP

텍스트 분류(Text Classification) 모델을 학습하기 전에 반드시 거쳐야 하는 과정이 **텍스트 전처리(Text Preprocessing)**입니다.  
언어 데이터의 특성상 불필요한 기호, 중복 표현, 대소문자 문제 등이 존재하기 때문에, 모델 성능을 높이기 위해 데이터를 정제하는 것이 매우 중요합니다.  

---

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from konlpy.tag import Okt  # 한국어 형태소 분석기

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# 예제 문장
text_en = "Apple is looking at buying U.K. startup for $1 billion! Running better?"
text_kr = "나는 학교에 갔다. 그리고 친구들을 만났다."

```

## 1. 기본 정제 (Cleaning)

- 불필요한 기호, 특수문자 제거  
  - 예: `, . ! ?` 같은 문장부호 제거
- 숫자 처리  
  - 그대로 둘지 / 토큰 `<NUM>`으로 대체할지 결정
- 대소문자 통일  
  - 영어의 경우 대부분 소문자 변환 (`Apple` → `apple`)

```python
# 영어
def clean_text_en(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r"[^a-z\s]", "", text)  # 알파벳과 공백만 남기기
    return text

print("EN Cleaning:", clean_text_en(text_en))

# 한국어
def clean_text_kr(text):
    text = re.sub(r"[^가-힣\s]", "", text)  # 한글과 공백만 남기기
    return text

print("KR Cleaning:", clean_text_kr(text_kr))
```

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

```python
# 영어: 단어 단위 토큰화
tokens_en = nltk.word_tokenize(clean_text_en(text_en))
print("EN Tokenization:", tokens_en)

# 한국어: 형태소 단위 토큰화
okt = Okt()
tokens_kr = okt.morphs(clean_text_kr(text_kr))
print("KR Tokenization:", tokens_kr)
```

---

## 3. 불용어 제거 (Stopword Removal)

분류에 큰 의미가 없는 단어 제거  

- 한국어: `은`, `는`, `이`, `가`, `그리고`, `하지만`  
- 영어: `is`, `the`, `and`, `but`  

```python
# 영어
stop_words_en = set(stopwords.words("english"))
filtered_en = [w for w in tokens_en if w not in stop_words_en]
print("EN Stopword Removal:", filtered_en)

# 한국어 (간단 예시)
stop_words_kr = ["은", "는", "이", "가", "그리고", "을", "를"]
filtered_kr = [w for w in tokens_kr if w not in stop_words_kr]
print("KR Stopword Removal:", filtered_kr)
```

---

## 4. 정규화 (Normalization)

- **어간 추출 (Stemming)**  
  - 단어의 원형으로 변환 (예: `running` → `run`)  
  - 단순 규칙 기반이라 부정확할 수 있음
- **표제어 추출 (Lemmatization)**  
  - 품사를 고려한 원형 변환 (`better` → `good`)  
- 한국어의 경우: 활용형 → 기본형 변환 (`갔다` → `가다`)  

```python
# 영어 - 어간 추출 (Stemming)
stemmer = PorterStemmer()
stemmed_en = [stemmer.stem(w) for w in filtered_en]
print("EN Stemming:", stemmed_en)

# 영어 - 표제어 추출 (Lemmatization)
lemmatizer = WordNetLemmatizer()
lemmatized_en = [lemmatizer.lemmatize(w) for w in filtered_en]
print("EN Lemmatization:", lemmatized_en)

# 한국어는 Okt의 'normalize' 또는 'stem' 옵션 사용 가능
tokens_kr_norm = okt.morphs(clean_text_kr(text_kr), stem=True)  # 활용형 → 기본형
print("KR Normalization:", tokens_kr_norm)
```

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
```
7. 전처리 예시 코드 (Python)
```python
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
```
## ✅ 정리

 - 텍스트 전처리는 다음 단계로 구성됩니다:

 - 기본 정제 → 특수문자, 숫자 처리, 대소문자 변환

 - 토큰화 → 단어/형태소/서브워드 단위 분할

 - 불용어 제거 → 의미 없는 단어 제거

 - 정규화 → 어간 추출, 표제어 추출

 - 인코딩 → BoW, TF-IDF, Word2Vec, BERT 토큰화

 - AutoTokenizer → Hugging Face 모델용 자동 토큰화, 패딩, 텐서 변환 지원

---
## 1. XAI가 필요한 이유

 - 블랙박스 문제: 딥러닝 모델은 높은 성능을 내지만, 의사결정 과정을 사람이 이해하기 어렵습니다.

 - 신뢰성: 의료, 금융, 법률, 자율주행 등 사회적으로 중요한 분야에서는 “왜 이 결과가 나왔는가?”를 설명할 수 있어야 합니다.

 - 규제와 법적 요구: EU AI Act, GDPR의 “자동화된 의사결정에 대한 설명권(Right to Explanation)” 같은 법적 배경이 있습니다.

 - 사용자 수용성: 설명이 있어야 사람들이 결과를 신뢰하고, AI를 실제로 활용할 수 있습니다.

## 2. XAI란 무엇인가?

 - 정의: 인공지능 모델의 의사결정 과정을 사람이 이해할 수 있는 형태로 설명하는 기술 또는 방법론.

 - 목표:

   - 모델의 투명성 (Transparency)

   - 모델의 해석 가능성 (Interpretability)

   - 사용자 신뢰 확보 (Trust)

## 3. XAI 접근 방식
### (1) 모델 중심 (Model-based)

 - 설명 가능한 모델 자체 사용

 - 의사결정나무(Decision Tree), 선형회귀, 규칙 기반 모델 등

 - 장점: 구조 자체가 직관적

 - 단점: 복잡한 데이터에서는 성능 한계

### (2) 사후 설명 (Post-hoc)

 - 블랙박스 모델을 유지하면서, 결과를 해석

 - LIME (Local Interpretable Model-agnostic Explanations): 특정 샘플 예측 근처를 선형 모델로 근사

 - SHAP (SHapley Additive exPlanations): 게임 이론 기반, 각 피처의 기여도 계산

 - Grad-CAM: CNN의 시각적 설명 (이미지에서 어떤 영역이 중요한지 Heatmap으로 표시)

## 4. XAI의 실제 활용 사례
- 의료: X-ray/CT에서 어떤 부위 때문에 질병이라고 진단했는지 시각적으로 표시

- 금융: 대출 심사에서 "소득, 직업, 신용점수" 중 어떤 요인이 승인/거절에 영향을 미쳤는지 설명

- 자율주행: 객체 탐지 모델이 어떤 도로 표지판이나 차량을 중요하게 판단했는지 시각화

## 5. XAI의 한계와 과제

- 정확성 vs 해석성: 설명 가능한 모델은 성능이 떨어질 수 있음

- 부분적 설명: LIME/SHAP은 국소적(Local) 설명을 제공, 전체 모델을 완벽히 설명하진 못함

- 사용자 수준의 차이: 전문가에게는 수학적/통계적 설명이 필요하고, 일반인에게는 직관적 시각화가 필요

## 6. XAI의 미래 방향

- Interpretable-by-design: 처음부터 설명 가능한 모델 설계

- Human-centered XAI: 사용자별 맞춤형 설명 (전문가용 vs 일반인용)

- 신뢰 + 공정성 + 투명성: AI 윤리와 함께 통합적으로 연구되는 방향
