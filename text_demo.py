import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

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
vector = vectorizer.transform([text1])
prediction = model.predict(vector)[0]
confidence = max(model.predict_proba(vector)[0])

label = "Fake" if prediction == 0 else "Real"

print("\n--- TEXT INPUT RESULT ---")
print("Prediction:", label)
print("Confidence:", round(confidence * 100, 2))

vector2 = vectorizer.transform([text2]) 
prediction2 = model.predict(vector2)[0]
confidence2 = max(model.predict_proba(vector2)[0])
label2 = "Fake" if prediction2 == 0 else "Real"
print("\n--- TEXT INPUT RESULT ---")
print("Prediction:", label2)
print("Confidence:", round(confidence2 * 100, 2))