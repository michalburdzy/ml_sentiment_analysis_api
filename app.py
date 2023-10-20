from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

app = FastAPI()

class TextInput(BaseModel):
  text: str

@app.get('/healthcheck')
async def healthcheck():
  return {"message": "success"}

@app.post('/predict')
async def scoring_endpoint(item: TextInput):
  tokens = tokenizer.encode(item.model_dump()['text'], return_tensors='pt')
  result = model(tokens)
  max_result = torch.argmax(result.logits)

  return {"score": int(max_result + 1)}