from fastapi import FastAPI
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# 데이터 가져오기 - 경로수정 필요
data = pd.read_excel('C:/Users/sally/Desktop/GDSC/question_producted_embedding.xlsx')
question = data['질문 (Query)'].tolist()
answer = data['답변 (Answer)'].tolist()

# 사전훈련한 임베딩 가져오기
embedding_tensor_data = torch.load('C:/Users/sally/Desktop/GDSC/ques_tensor.pt')

@app.get("/{text}")
async def load_text(prompt: str):
    return answering(prompt)

def answering(prompt):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # 예시 문장 공백 제거하기
    prompt = prompt.replace(' ', '')

    # 예시 문장 인코딩하기
    prompt = model.encode(prompt)

    # Tensor
    prompt_to_tensor = torch.tensor(prompt)

    # 유사도 측정하기
    cosine_scores = util.pytorch_cos_sim(prompt_to_tensor, embedding_tensor_data)

    # 출력
    # 가장 유사도가 높은 인덱스 가져오기
    top_similarity = cosine_scores[0].argsort(descending=True)[0]

    # 답변 return 해주기
    return answer[top_similarity]