import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------
# IMPORTANT: import objects from data_processing
# ------------------------------------------------
from data_processing import (
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test,
    tfidf
)

# ------------------------------------------------
# STEP 1: Train the model
# ------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ------------------------------------------------
# STEP 2: Evaluate the model
# ------------------------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------
# STEP 3: Explainability (XAI)
# Show top words influencing prediction
# ------------------------------------------------
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# Top words for REAL news
top_real_indices = np.argsort(coefficients)[-10:]
top_real_words = feature_names[top_real_indices]

# Top words for FAKE news
top_fake_indices = np.argsort(coefficients)[:10]
top_fake_words = feature_names[top_fake_indices]

print("\nTop words indicating REAL news:")
print(top_real_words)

print("\nTop words indicating FAKE news:")
print(top_fake_words)

# ------------------------------------------------
# STEP 4: Save model and vectorizer
# ------------------------------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
