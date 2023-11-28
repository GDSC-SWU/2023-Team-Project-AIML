import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#nltk.download('punkt')



app = FastAPI()
model = SentenceTransformer('snunlp/KR-Sbert-V40K-klueNLI-augSTS')
embedding_data = torch.load("C:/Users/thdek/Downloads/embedding_data.pt")
df = pd.read_excel("C:/Users/thdek/Downloads/Qusetion.xlsx")


@app.get("/test/{text}")
async def get_answer(original_text: str):
    return get_answer(original_text)


def get_answer(sample):

    sample_encode = model.encode(sample)
    sample_tensor = torch.tensor(sample_encode)

    cos_sim = util.cos_sim(sample_tensor, embedding_data)

    best_sim_idx = int(np.argmax(cos_sim))

    result = df['답변 (Answer)'][best_sim_idx]
    return result


