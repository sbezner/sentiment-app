from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the smaller 3-class model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, phrase: str = Form(...)):
    prediction = classifier(phrase)[0]
    label = prediction["label"].lower()  # Already "positive", "neutral", or "negative"
    score = prediction["score"]
    score_percent = f"{score * 100:.1f}%"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": label,
        "phrase": phrase,
        "score": score_percent
    })
