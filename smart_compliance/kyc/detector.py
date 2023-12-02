from deepface import DeepFace
import base64
import cv2
import numpy as np


def detect_faces(image):
    try:
        backends = [
            'opencv',
            'ssd',
            'dlib',
            'mtcnn',
            'retinaface',
            'mediapipe',
            'yolov8',
            'yunet',
            'fastmtcnn',
        ]
        return DeepFace.extract_faces(img_path=image,
                                      target_size=(224, 224),
                                      detector_backend=backends[4])
    except Exception as e:
        # TODO: this error seems to break the app completely. Fix it.
        # RuntimeError: Event loop is closed
        print("Could not detect faces.")
        return []


def extract_faces_for_selfie(image):
    faces = []
    for face in detect_faces(image):
        facial_area = face["facial_area"]
        w = facial_area["w"]
        h = facial_area["h"]
        size = w * h
        faces.append({
            "face": facial_area,
            "size": size,
        })
    backup = np.array(image).copy()
    detected_faces = []
    faces.sort(key=lambda face: face["size"], reverse=True)
    for face in list(map(lambda face: face["face"], faces[:2])):
        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]
        detected_faces.append(backup[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return detected_faces, image


def extract_faces_for_document(image):
    faces = detect_faces(image)
    face = faces[0] if len(faces) > 0 else None
    if face:
        facial_area = face["facial_area"]
        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return [face], image


def verify_face(base_image, image, models, distance_metrics, detector_backend="opencv"):
    if base_image is None or image is None:
        return []

    results = []
    for model in models:
        for distance_metric in distance_metrics:
            result = DeepFace.verify(img1_path=base_image,
                                     img2_path=image,
                                     model_name=model,
                                     distance_metric=distance_metric,
                                     detector_backend=detector_backend,
                                     enforce_detection=False)
            results.append(result)

    results = [{
        "base": f"data:image/jpg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))[1]).decode('utf-8')}",
        "face": f"data:image/jpg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[1]).decode('utf-8')}",
        "verified": result['verified'],
        "model": result['model'],
        "similarity_metric": result['similarity_metric'],
        "distance": result['distance'],
        "threshold": result['threshold'],
    } for result in results]
    return results


def get_prediction(results, model_weights):
    verified = []
    score = 0
    for result in results:
        if result["verified"]:
            score += sum([mw['weight'] /
                         3 for mw in model_weights if mw['model'] == result["model"]])
            verified.append(result)

    # score = score * 100 / sum([mw['weight'] for mw in model_weights])
    score_formatted = "{:.2f}".format(score)
    if score >= 50:
        return "verified", score_formatted

    return "not_verified", score_formatted
