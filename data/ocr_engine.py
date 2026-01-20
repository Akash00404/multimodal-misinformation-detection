import pytesseract
import cv2
from PIL import Image
import os

# macOS Tesseract path
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or invalid image path")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return processed


def extract_text_from_image(image_path, lang="eng"):
    """
    Extract text from image using Tesseract OCR
    """
    processed_image = preprocess_image(image_path)

    text = pytesseract.image_to_string(
        processed_image,
        lang=lang,
        config="--psm 6"
    )

    return text.strip()


def is_valid_ocr_text(text):
    """
    Validate OCR output
    """
    if not text:
        return False
    if len(text.strip()) < 20:
        return False
    return True
