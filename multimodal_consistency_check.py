import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Load TF-IDF Vectorizer
# ---------------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------------
# Multimodal Consistency Function
# ---------------------------------
def check_multimodal_consistency(text_1, text_2, threshold=0.3):
    """
    text_1: Article / Caption text
    text_2: Image OCR text
    threshold: similarity cutoff
    """

    vectors = tfidf.transform([text_1, text_2])
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    if similarity_score < threshold:
        status = "Inconsistent (Potentially Misleading)"
    else:
        status = "Consistent"

    return {
        "Similarity Score": round(similarity_score, 3),
        "Consistency Status": status
    }


# ---------------------------------
# Example Test
# ---------------------------------
if __name__ == "__main__":

    article_text = """
    The government announced new economic reforms to boost the manufacturing sector.
    """

    image_text = """
    This image shows an unrelated protest from several years ago.
    """

    result = check_multimodal_consistency(article_text, image_text)

    print("\n--- Multimodal Consistency Check ---")
    for key, value in result.items():
        print(f"{key}: {value}")
