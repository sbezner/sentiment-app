import datetime as dt
from flask import Flask, request, render_template
import praw
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from asgiref.wsgi import WsgiToAsgi
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

assert reddit.read_only

def analyze_ticker(ticker, hours):
    since = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    posts, comments = [], []

    for submission in reddit.subreddit("all").search(f"title:{ticker}", sort="new", limit=200):
        created = dt.datetime.utcfromtimestamp(submission.created_utc)
        if created < since: continue
        posts.append(analyzer.polarity_scores(submission.title)["compound"])
        submission.comments.replace_more(limit=0)
        for c in submission.comments:
            if dt.datetime.utcfromtimestamp(c.created_utc) >= since:
                comments.append(analyzer.polarity_scores(c.body)["compound"])

    def summarize(scores):
        return {
            "count": len(scores),
            "mean": sum(scores)/len(scores) if scores else 0.0,
            "pos": sum(1 for s in scores if s > 0.05),
            "neu": sum(1 for s in scores if -0.05 <= s <= 0.05),
            "neg": sum(1 for s in scores if s < -0.05),
        }

    ps = summarize(posts)
    cs = summarize(comments)
    rec = "Yes" if ps["mean"] >= 0.1 and ps["count"] >= 3 else "No"

    info = yf.Ticker(ticker).history(period="2d")
    change = 0.0
    if len(info) >= 2:
        change = (info['Close'][-1] / info['Close'][-2] - 1) * 100

    return ps, cs, rec, round(change, 2)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    hours = request.form.get("hours", 8, type=int)
    tickers_input = request.form.get("tickers", "").upper()
    if tickers_input:
        raw = tickers_input.replace(",", "\n").splitlines()
        tickers = [t.strip() for t in raw if t.strip()]
        temp_results = []

        for t in tickers:
            ps, cs, rec, pct = analyze_ticker(t, hours)
            score = ps["mean"] * ps["count"] + pct
            temp_results.append((t, {
                "posts": ps,
                "comments": cs,
                "recommend": rec,
                "change": pct,
                "score": score
            }))

        # Sort and attach rank
        temp_results.sort(key=lambda x: x[1]["score"], reverse=True)
        results = []
        for rank, (t, data) in enumerate(temp_results, start=1):
            data["rank"] = rank
            data["ticker"] = t
            results.append(data)

    return render_template("index.html", results=results, hours=hours)

wsgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    app.run(debug=True)
