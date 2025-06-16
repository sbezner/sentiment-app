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
    label = prediction["label"]
    score = prediction["score"]

    # Map model labels to human-friendly strings
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    result = label_map.get(label.upper(), label)  # Fallback to raw label
    score_percent = f"{score * 100:.1f}%"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "phrase": phrase,
        "score": score_percent
    })

