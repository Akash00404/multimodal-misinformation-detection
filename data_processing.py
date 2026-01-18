import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
fake_df = pd.read_csv("data/raw/Fake.csv")
true_df = pd.read_csv("data/raw/True.csv")

# -----------------------------
# STEP 2: Add labels
# Fake = 0, Real = 1
# -----------------------------
fake_df["label"] = 0
true_df["label"] = 1

# -----------------------------
# STEP 3: Combine datasets
# -----------------------------
data = pd.concat([fake_df, true_df], axis=0)
data = data.reset_index(drop=True)

# -----------------------------
# STEP 4: Select text column
# -----------------------------
data = data[["text", "label"]]

# -----------------------------
# STEP 5: Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()

data["text"] = data["text"].apply(clean_text)

# -----------------------------
# STEP 6: Split features & labels
# -----------------------------
X = data["text"]
y = data["label"]

# -----------------------------
# STEP 7: Train-test split (80-20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# STEP 8: TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# STEP 9: Final confirmation prints
# -----------------------------
print("Data processing completed successfully!")
print("Training samples:", X_train_tfidf.shape)
print("Testing samples:", X_test_tfidf.shape)
