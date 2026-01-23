import joblib
from textblob import TextBlob

from data.url_scraper import extract_text_from_url, is_valid_url_text
from trust_score_engine import calculate_trust_score
from source_reliability import get_source_reliability


model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def analyze_url(url):
    """
    Analyze news article from URL
    """
    article_text = extract_text_from_url(url)

    if not is_valid_url_text(article_text):
        return {
            "error": "Insufficient article text extracted"
        }

    vector = vectorizer.transform([article_text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    label = "Fake" if prediction == 0 else "Real"

    sentiment_score = TextBlob(article_text).sentiment.polarity

    trust_score = calculate_trust_score(
        model_confidence=confidence,
        sentiment_score=sentiment_score,
        source_score = get_source_reliability(url) # URLs assumed higher reliability
    )

    return {
        "prediction": label,
        "confidence_percent": round(confidence * 100, 2),
        "trust_score_percent": trust_score
    }

if __name__ == "__main__":
    url = "https://www.bbc.com/news/world-europe-60506682"

    result = analyze_url(url)

    print("\n--- URL ANALYSIS RESULT ---")
    for key, value in result.items():
        print(f"{key}: {value}")
