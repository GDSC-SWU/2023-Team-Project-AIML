from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

model = SentenceTransformer('snunlp/KR-Sbert-V40K-klueNLI-augSTS')
embedding_data = torch.load("C://Users//정은지//Downloads//embedding_data.pt")
df = pd.read_excel("C://Users//정은지//Downloads//Qusetion.xlsx")

@app.get("/chatbot/{text}")
async def chatbot(original_text : str):
    return {chatbot(original_text)}

def chatbot(sample):

    #공백 제거
    sample = sample.replace(" ","")

    #질문 예시 문장 인코딩 후 텐서화
    sample_encode = model.encode(sample)
    sample_tensor = torch.tensor(sample_encode)

    #저장한 임베딩 데이터와의 코사인 유사도 측정
    cos_sim = util.cos_sim(sample_tensor, embedding_data)

    # #가장 높은 코사인 유사도 가지는 질문 인덱스 출력
    # print(f"가장 높은 코사인 유사도 idx : {int(np.argmax(cos_sim))}")

    #선택된 질문 출력
    best_sim_idx = int(np.argmax(cos_sim))
    selected_qes = df['질문 (Query)'][best_sim_idx]

    #선택된 질문 문장에 대한 인코딩 진행
    selected_qes_encode = model.encode(selected_qes)

    #선택한 질문과의 유사도 점수 측정
    score = np.dot(sample_tensor, selected_qes_encode) / (
                np.linalg.norm(sample_tensor) * np.linalg.norm(selected_qes_encode))

    #선택된 질문에 대한 답변
    answer = df['답변 (Answer)'][best_sim_idx]
    return answer
