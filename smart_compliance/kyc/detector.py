from deepface import DeepFace
import base64
import cv2


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
        return DeepFace.extract_faces(img_path = image,
            target_size = (224, 224),
            detector_backend = backends[4]
        )
    except Exception as e:
        print("Exception: ", e)
        return []

def with_rectangles(image):
    faces = detect_faces(image)

    for face in faces:
        try:
            facial_area = face["facial_area"]
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        except Exception as e:
            print("Problem drawing face: " + str(e))

    return image


def extract_faces(image):
    faces = []
    for face in detect_faces(image):
        faces.append(face["face"])
    return faces


def verify_face(base_image, image, models, distance_metrics, detector_backend="opencv"):
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

