#!/usr/bin/env python3
import requests
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse

# === Argparse Setup ===
parser = argparse.ArgumentParser(
    description="Fetch Reddit sentiment for a given stock ticker over the past X hours."
)
parser.add_argument(
    "-t", "--ticker",
    type=str,
    required=True,
    help="Stock ticker to analyze (e.g., TSLA)"
)
parser.add_argument(
    "-H", "--hours",
    type=int,
    default=8,
    help="Time window in hours (default: 8)"
)

args = parser.parse_args()
TICKER = args.ticker
HOURS = args.hours

# === Config ===
SUBREDDITS = [TICKER, TICKER + "Stock", TICKER.upper()]
KEYWORD = TICKER
MAX_RESULTS = 500

# === Timestamp Setup ===
now = dt.datetime.utcnow()
start_ts = int((now - dt.timedelta(hours=HOURS)).timestamp())

# === Pushshift Query ===
def fetch_pushshift(kind="submission"):
    url = f"https://api.pushshift.io/reddit/{kind}/search"
    params = {
        "q": KEYWORD,
        "subreddit": ",".join(SUBREDDITS),
        "after": start_ts,
        "size": MAX_RESULTS,
        "sort": "desc"
    }
    return requests.get(url, params=params).json().get("data", [])

posts = fetch_pushshift("submission")
comments = fetch_pushshift("comment")

# === Sentiment Analysis ===
analyzer = SentimentIntensityAnalyzer()

def analyze(items, key):
    scores = [analyzer.polarity_scores(item.get(key, ""))["compound"] for item in items]
    return {
        "count": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "pos": sum(s > 0.05 for s in scores),
        "neu": sum(-0.05 <= s <= 0.05 for s in scores),
        "neg": sum(s < -0.05 for s in scores)
    }

post_sent = analyze(posts, "title")
comment_sent = analyze(comments, "body")

# === Results ===
print(f"\n🔍 {TICKER} sentiment in the past {HOURS} hours:")
print(f"Submissions: count={post_sent['count']}, mean={post_sent['mean_score']:+.3f}, +={post_sent['pos']} n={post_sent['neu']} -={post_sent['neg']}")
print(f"Comments:    count={comment_sent['count']}, mean={comment_sent['mean_score']:+.3f}, +={comment_sent['pos']} n={comment_sent['neu']} -={comment_sent['neg']}\n")

