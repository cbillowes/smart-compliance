import cv2
from domain.denoisify import denoisify
from domain.rotate import rotate
from domain.detector import detect_faces

def prepare(image_path):
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = denoisify(original_image)
    image = rotate(image, angle=45,
                   loop_while=lambda x: len(detect_faces(x, scale_factor=1.1)) == 0)
    return (original_image, image)
