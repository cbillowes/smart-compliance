import cv2
from domain.rotate import rotate
from domain.detector import detect_faces
import matplotlib.pyplot as plt

def detected_faces(image):
    return len(detect_faces(image)) > 0


def prepare(image_path):
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = rotate(original_image, angle=45,
                   loop_while=lambda x: detected_faces(x) == 0)
    if (detected_faces(image) == 0):
        raise Exception(f"No faces detected in {image_path}")

    return (original_image, image)
