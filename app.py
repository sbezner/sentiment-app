from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, phrase: str = Form(...)):
    prediction = classifier(phrase)[0]
    label = prediction["label"].lower()
    score = prediction["score"]

    if 0.45 < score < 0.55:
        result = "neutral"
    else:
        result = "positive" if "pos" in label else "negative"

    # Round score for display
    score_percent = f"{score * 100:.1f}%"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "phrase": phrase,
            "score": score_percent
        }
    )
