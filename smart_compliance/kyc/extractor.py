import cv2
import pytesseract
from PIL import Image
import re


def preprocess_image(image):
    """
    Processes the image by resizing it in order to improve OCR results.
    """
    return cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)


def extract_words_from_text(text, word):
    """
    Extracts all words from the given text that match the given word.
    """
    return " ".join(re.findall(word, text))


def extract_text(image, personal_details):
    """
    Extracts text from the given image using pytesseract.
    """
    try:
        print("Extracting text...")
        pil_image = Image.fromarray(image)
        config = '--psm 6'
        text = pytesseract.image_to_string(pil_image, config=config).lower()

        return {
            "raw_text": text,
            "first_name": extract_words_from_text(text, personal_details["first_name"].lower()),
            "last_name": extract_words_from_text(text, personal_details["last_name"].lower()),
            "id_number": extract_words_from_text(text, personal_details["id_number"].lower()),
            "dob": extract_words_from_text(text, personal_details["dob"].lower()),
        }
    except Exception as e:
        print(f"Could not extract text: {e}")
