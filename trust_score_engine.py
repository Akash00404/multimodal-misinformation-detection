import joblib
import numpy as np
from textblob import TextBlob

# -----------------------------
# STEP 1: Load trained model
# -----------------------------
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# STEP 2: Helper functions
# -----------------------------

def get_sentiment_score(text):
    """
    Returns:
    sentiment_label: Positive / Neutral / Negative
    sentiment_stability_score: value between 0 and 1
    """
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive", 0.6
    elif polarity < -0.1:
        return "Negative", 0.4
    else:
        return "Neutral", 1.0


def calculate_trust_score(model_confidence, sentiment_score, source_score=0.5):
    """
    Trust Score formula (review-safe & explainable)
    """
    trust_score = (
        0.5 * model_confidence +
        0.3 * source_score +
        0.2 * sentiment_score
    )
    return round(trust_score * 100, 2)


# -----------------------------
# STEP 3: Analyze input text
# -----------------------------
def analyze_news(text):
    # Vectorize text
    text_vec = tfidf.transform([text])

    # Prediction
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    model_confidence = max(probabilities)
    label = "Real" if prediction == 1 else "Fake"

    # Sentiment analysis
    sentiment_label, sentiment_score = get_sentiment_score(text)

    # Source reliability (placeholder for now)
    source_reliability_score = 0.5  # neutral (will be dynamic for URLs later)

    # Trust Score
    trust_score = calculate_trust_score(
        model_confidence,
        sentiment_score,
        source_reliability_score
    )

    return {
        "Prediction": label,
        "Model Confidence (%)": round(model_confidence * 100, 2),
        "Sentiment": sentiment_label,
        "Trust Score (%)": trust_score
    }


# -----------------------------
# STEP 4: Run example
# -----------------------------
if __name__ == "__main__":
    sample_text = """
    Breaking news! Shocking claims are spreading rapidly across social media.
    This information has not been verified by any official sources.
    """

    result = analyze_news(sample_text)

    print("\n--- Analysis Result ---")
    for key, value in result.items():
        print(f"{key}: {value}")
