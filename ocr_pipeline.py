import joblib

from data.ocr_engine import extract_text_from_image, is_valid_ocr_text
from trust_score_engine import calculate_trust_score
from multimodal_consistency_check import check_multimodal_consistency
from textblob import TextBlob



# Load trained ML components
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def analyze_image(image_path, caption_text=None):
    """
    Complete OCR-based misinformation analysis
    """
    # Step 1: Extract text using OCR
    ocr_text = extract_text_from_image(image_path)

    if not is_valid_ocr_text(ocr_text):
        return {
            "error": "OCR text insufficient for analysis"
        }

    # Step 2: Vectorize text
    vector = vectorizer.transform([ocr_text])

    # Step 3: Predict Fake / Real
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    label = "Fake" if prediction == 0 else "Real"
    # Step 4: Trust Score
    # Calculate sentiment polarity
    sentiment_score = TextBlob(ocr_text).sentiment.polarity

    # Calculate trust score using correct parameters
    trust_score = calculate_trust_score(
        model_confidence=confidence,
        sentiment_score=sentiment_score,
        source_score=0.7   # Images assumed moderate reliability
    )

    # Step 5: Multimodal Consistency (if caption exists)
    consistency_result = None
    if caption_text:
        similarity, status = check_multimodal_consistency(ocr_text, caption_text)
        consistency_result = {
            "similarity_score": similarity,
            "status": status
        }

    return {
        "ocr_text": ocr_text,
        "prediction": label,
        "confidence_percent": round(confidence * 100, 2),
        "trust_score_percent": trust_score,
        "consistency": consistency_result
    }

if __name__ == "__main__":
    image_path = "test_images/test1.png"

    result = analyze_image(image_path)

    print("\n--- OCR ANALYSIS RESULT ---")
    for key, value in result.items():
        print(f"{key}: {value}")
