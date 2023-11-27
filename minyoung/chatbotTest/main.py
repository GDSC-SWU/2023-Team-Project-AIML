import torch
import numpy as np
import pandas as pd

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util

app = FastAPI()


@app.get("/chatbot/{text}")
async def chat(sentence: str):
    return chatbot(sentence)

def chatbot(sentence):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    embedding_data = torch.load(r"C:\Users\rucy0\Downloads\embedding_data.pt")
    df = pd.read_excel(r"C:\Users\rucy0\Downloads\Qusetion.xlsx")
    # 사용자로부터 질문 입력 받기
    question = sentence

    # 입력 받은 질문 출력
    print("입력한 질문:", question)

    # 입력 받은 질문 공백 제거
    question = question.replace(" ", "")
    print("공백 제거 문장: ", question)

    # 질문 인코딩 후 텐서화
    question_encode = model.encode(question)
    question_tensor = torch.tensor(question_encode)

    # 코사인 유사도 계산
    cos_sim = util.cos_sim(question_tensor, embedding_data)

    # 유사도가 가장 높은 질문의 인덱스 추출
    best_sim_idx = int(np.argmax(cos_sim))
    selected_qes = df['질문 (Query)'][best_sim_idx]

    selected_qes_encode = model.encode(selected_qes)

    # 해당 인덱스의 답변 돌려주기
    answer = df['답변 (Answer)'][best_sim_idx]
    return answer
