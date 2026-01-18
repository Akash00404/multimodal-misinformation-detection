import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load LIAR dataset
liar_df = pd.read_csv(
    "data/liar/test.tsv",
    sep="\t",
    header=None
)

# Columns: label, statement
liar_df = liar_df[[1, 2]]
liar_df.columns = ["label", "text"]

# Proper binary mapping (standard practice)
def map_label(label):
    if label in ["true", "mostly-true"]:
        return 1
    elif label in ["false", "barely-true", "pants-fire"]:
        return 0
    else:
        return None  # remove half-true

liar_df["label"] = liar_df["label"].apply(map_label)
liar_df = liar_df.dropna()

# Balance the dataset
real_df = liar_df[liar_df["label"] == 1]
fake_df = liar_df[liar_df["label"] == 0]

min_size = min(len(real_df), len(fake_df))
liar_balanced = pd.concat([
    real_df.sample(min_size, random_state=42),
    fake_df.sample(min_size, random_state=42)
])

# Vectorize
X = tfidf.transform(liar_balanced["text"])
y = liar_balanced["label"]

# Predict
y_pred = model.predict(X)

# Evaluate
print("\n--- LIAR DATASET VALIDATION (CLEAN & BALANCED) ---")
print("Samples used:", len(liar_balanced))
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
