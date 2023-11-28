from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('snunlp/KR-Sbert-V40K-klueNLI-augSTS')
embedding_data = torch.load('embedding_data.pt')
df = pd.read_excel('Question_email.xlsx')


app = FastAPI()


@app.get("/chatbot/{ans}")
async def chat(sentence: str):
    return chatbot(sentence)


def chatbot(sentence):
    sentence = sentence.replace(" ", "")

    sentence_encode = model.encode(sentence)
    sentence_tensor = torch.tensor(sentence_encode)

    cos_sim = util.cos_sim(sentence_tensor, embedding_data)

    best_sim_idx = int(np.argmax(cos_sim))
    selected_qes = df['질문 (Query)'][best_sim_idx]

    selected_qes_encode = model.encode(selected_qes)

    score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
    print(score)

    if score < 0.5:
        answer = "죄송합니다 알맞은 답변을 찾지 못했습니다 :)"
        answer_email = ""
    else:
        answer = df['답변 (Answer)'][best_sim_idx]
        answer_email = df['이메일 (email)'][best_sim_idx]

    return answer, answer_email
