# app.py
import datetime as dt
from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# === Test data simulating sentiment scores ===
TEST_DATA = {
    'TSLA': {'posts': [0.3, 0.1, -0.1], 'comments': [0.2, -0.05, 0.0, 0.5]},
    'AAPL': {'posts': [0.0, 0.05], 'comments': [-0.1, 0.0, 0.1]},
    # Add more tickers here if needed
}

def analyze_ticker(ticker, hours):
    data = TEST_DATA.get(ticker, {'posts': [], 'comments': []})
    def summarize(lst):
        return {
            "count": len(lst),
            "mean": sum(lst) / len(lst) if lst else 0.0,
            "pos": sum(1 for s in lst if s > 0.05),
            "neu": sum(1 for s in lst if -0.05 <= s <= 0.05),
            "neg": sum(1 for s in lst if s < -0.05),
        }
    return summarize(data['posts']), summarize(data['comments'])

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    hours = request.form.get("hours", 8, type=int)
    tickers_input = request.form.get("tickers", "").upper()
    if tickers_input:
        # Split input by commas or newlines, strip whitespace
        raw_list = tickers_input.replace(',', '\n').splitlines()
        tickers = [t.strip() for t in raw_list if t.strip()]
        results = {}
        for t in tickers:
            ps, cs = analyze_ticker(t, hours)
            results[t] = {"posts": ps, "comments": cs}
    return render_template("index.html", results=results, hours=hours)

# Wrap WSGI app for ASGI compatibility
from asgiref.wsgi import WsgiToAsgi
wsgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    app.run(debug=True)
