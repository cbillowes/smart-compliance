import cv2
from domain.rotate import rotate
from domain.detector import detect_faces
import matplotlib.pyplot as plt


def detected_faces(image):
    return len(detect_faces(image)) > 0


def process(image, scale_factor=1.1):
    processed = cv2.resize(
        image, (0, 0), fx=scale_factor, fy=scale_factor)
    processed = rotate(processed, angle=10,
                       loop_while=lambda x: detected_faces(x) == 0)
    return (image, processed)
