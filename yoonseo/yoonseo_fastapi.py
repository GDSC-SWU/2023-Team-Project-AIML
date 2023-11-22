from fastapi import FastAPI

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

embedding_data=torch.load('/Users/2ynnso/GDSC_toy_chatbot_api/embedding_data.pt')
df=pd.read_excel('/Users/2ynnso/GDSC_toy_chatbot_api/trian_Data_embedding.xlsx')

app = FastAPI()

@app.get("/chatbot/{text}")
async def chatbot(question : str):
    return {chatbot(question)}

def chatbot(sample):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    embedding_data = torch.load('/Users/2ynnso/GDSC_toy_chatbot_api/embedding_data.pt')
    df = pd.read_excel('/Users/2ynnso/GDSC_toy_chatbot_api/trian_Data_embedding.xlsx')

    sentence=sample

    sentence=sentence.replace(" ","") #공백 제거하기

    # 질문 예시 문장 인코딩 후 텐서화
    sentence_encode = model.encode(sentence)  # 인코딩
    sentence_tensor = torch.tensor(sentence_encode)  # 텐서화

    # 저장한 임베딩 데이터와의 코사인 유사도 측정
    cos_sim = util.cos_sim(sentence_tensor, embedding_data)

    # 선택된 질문 출력
    best_sim_idx = int(np.argmax(cos_sim))
    selected_qes = df['질문 (Query)'][best_sim_idx]

    # 선택된 질문 문장에 대한 인코딩
    selected_qes_encode = model.encode(selected_qes)

    # 유사도 점수 측정
    score = np.dot(sentence_tensor, selected_qes_encode) / (
                np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))

    answer = df['답변 (Answer)'][best_sim_idx]
    return answer