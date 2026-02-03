# Multimodal Misinformation Detection

End-to-end pipeline for detecting misinformation across text, URLs, and images. The system combines a TF-IDF + Logistic Regression baseline, a fine-tuned BERT classifier, OCR for image text, URL scraping for article text, a multimodal consistency check, and a composite trust score.

## What This Project Does

This repository provides:
- A supervised text classifier trained on Fake/True datasets (TF-IDF + Logistic Regression).
- A BERT-based classifier for semantic prediction.
- OCR pipeline to extract text from images.
- URL pipeline to extract article text from news links.
- Multimodal consistency checking between caption/article text and OCR text.
- A trust score that blends model confidence, sentiment polarity, and source reliability.

## Pipeline Overview

1. Data preparation:
   - Load `data/raw/Fake.csv` and `data/raw/True.csv`.
   - Clean text, label, split, and vectorize with TF-IDF.
   - Artifacts: `fake_news_model.pkl`, `tfidf_vectorizer.pkl`.

2. Baseline model training + explainability:
   - Train Logistic Regression on TF-IDF vectors.
   - Report accuracy and top indicator words for real/fake.

3. BERT model:
   - Train a BERT sequence classifier on the same Fake/True data.
   - Save fine-tuned model to `bert_module/bert_fake_news_model/`.

4. Input pipelines:
   - Text: direct TF-IDF + BERT predictions (see `text_demo.py`).
   - URL: scrape article text, validate length, predict, compute trust.
   - Image: OCR -> validate text -> predict -> trust score -> optional consistency check.

5. Trust scoring:
   - Combine model confidence, sentiment polarity, and source reliability into a single score.

6. Multimodal consistency:
   - Compare OCR text and caption/article text via TF-IDF cosine similarity.
   - Flag inconsistent pairs as potentially misleading.

## Key Files

- `data_processing.py`:
  - Loads and cleans Fake/True datasets.
  - Splits train/test and builds TF-IDF vectors.

- `model_training_xai.py`:
  - Trains Logistic Regression baseline.
  - Prints metrics + top real/fake indicator words.
  - Saves `fake_news_model.pkl` and `tfidf_vectorizer.pkl`.

- `bert_module/bert_training.py`:
  - Trains BERT classifier.
  - Saves model to `bert_module/bert_fake_news_model/`.

- `bert_module/bert_pipeline.py`:
  - Loads BERT model and runs inference.

- `trust_score_engine.py`:
  - Computes trust score from model confidence, sentiment, and source reliability.

- `source_reliability.py`:
  - Domain-based reliability lookup for common news sources.

- `data/url_scraper.py` + `url_pipeline.py`:
  - Extract article text from a URL and run predictions.

- `data/ocr_engine.py` + `ocr_pipeline.py`:
  - Extract text from an image via OCR and run predictions.

- `multimodal_consistency_check.py`:
  - Compares OCR text with caption/article text.

- `liar_validation.py`:
  - Evaluates model on the LIAR dataset (balanced subset).

## Quick Start (Typical Flow)

1. Prepare data and vectorizer:
   - Run `python data_processing.py`

2. Train baseline model:
   - Run `python model_training_xai.py`

3. (Optional) Train BERT model:
   - Run `python bert_module/bert_training.py`

4. Analyze:
   - Text: `python text_demo.py`
   - URL: `python url_pipeline.py`
   - Image: `python ocr_pipeline.py`

## Notes

- OCR expects Tesseract installed at `/opt/homebrew/bin/tesseract` on macOS.
- URL extraction uses `newspaper` to parse full article text.
- Trust score is a heuristic intended to be explainable and review-safe.

## Dataset Locations

- Fake/True: `data/raw/Fake.csv`, `data/raw/True.csv`
- LIAR dataset: `data/liar/test.tsv`
