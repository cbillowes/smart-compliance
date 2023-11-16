import cv2
import pytesseract
from PIL import Image

# https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/
def extract_characters(image):
    image = cv2.resize(image, (0, 0), fx=10, fy=10)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(Image.fromarray(threshold))
    return text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
