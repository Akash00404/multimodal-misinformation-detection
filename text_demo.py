import joblib

# ---- TF-IDF MODEL ----
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---- BERT MODEL ----
from bert_module.bert_pipeline import bert_predict


text1 = """
The Indian government on Monday announced a new initiative aimed at improving digital literacy in rural areas. 
According to officials from the Ministry of Electronics and Information Technology, the program will provide 
training in basic computer skills, online safety, and digital payments. The initiative will be implemented 
in collaboration with state governments and local educational institutions. Officials stated that the program 
is expected to benefit over five million citizens in its first phase. The ministry also clarified that no 
direct cash transfers are involved in the scheme.
"""

text2 = """
The government has secretly installed microchips inside all newly issued currency notes to track citizens’ spending habits. 
According to viral social media posts, these chips can transmit location data even when the notes are buried underground. 
Experts on messaging platforms claim this technology is powered by satellite signals and does not require any external power source. 
Officials have allegedly refused to comment on the matter, raising concerns about privacy and surveillance among citizens.
"""


def tfidf_predict(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    label = "Fake" if prediction == 0 else "Real"
    return label, round(confidence * 100, 2)


# -------- TEXT 1 --------
print("\n====== TEXT 1 ======")

tfidf_label, tfidf_conf = tfidf_predict(text1)
bert_result_1 = bert_predict(text1)

print("TF-IDF Prediction:", tfidf_label)
print("TF-IDF Confidence:", tfidf_conf)

print("BERT Prediction:", bert_result_1["prediction"])
print("BERT Confidence:", bert_result_1["confidence_percent"])


# -------- TEXT 2 --------
print("\n====== TEXT 2 ======")

tfidf_label2, tfidf_conf2 = tfidf_predict(text2)
bert_result_2 = bert_predict(text2)

print("TF-IDF Prediction:", tfidf_label2)
print("TF-IDF Confidence:", tfidf_conf2)

print("BERT Prediction:", bert_result_2["prediction"])
print("BERT Confidence:", bert_result_2["confidence_percent"])
