from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


def detect_faces(image,
                 cascade_path="haarcascade_frontalface_default.xml",
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30)):
    """
    https://github.com/kipr/opencv/tree/master/data/haarcascades
    """
    file = f"{cv2.data.haarcascades}{cascade_path}"
    face_cascade = cv2.CascadeClassifier(file)
    faces = face_cascade.detectMultiScale(image,
                                          scaleFactor=scale_factor,
                                          minNeighbors=min_neighbors,
                                          minSize=min_size)
    return faces


def with_rectangles(image,
                    cascade_path="haarcascade_frontalface_default.xml",
                    scale_factor=1.1,
                    min_neighbors=5,
                    min_size=(30, 30)):
    faces = detect_faces(image,
                         cascade_path=cascade_path,
                         scale_factor=scale_factor,
                         min_neighbors=min_neighbors,
                         min_size=min_size)

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return image


def extract_faces(image, scale_factor=1):
    faces = []
    for i, face in enumerate(detect_faces(image)):
        (x, y, w, h) = face
        rect = image[y:y + h, x:x + w]
        resized = cv2.resize(rect, (h * scale_factor, w * scale_factor))
        faces.append(resized)
    return faces


def verify_faces(selfie, document):
    faces = detect_faces(selfie)

    for i, face in enumerate(faces):
        (x, y, w, h) = face
        selfie = selfie[y:y + h, x:x + w]

    # preview_faces(selfie)
    # preview_faces(document)
    # return DeepFace.verify(selfie, document, model_name="Facenet", distance_metric="euclidean_l2", detector_backend="opencv")
