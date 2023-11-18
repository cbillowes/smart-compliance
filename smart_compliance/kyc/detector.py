from deepface import DeepFace
import cv2


def detect_faces(image,
                 cascade_path="haarcascade_frontalface_default.xml",
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30)):
    """
    https://github.com/kipr/opencv/tree/master/data/haarcascades
    """
    try:
        file = f"{cv2.data.haarcascades}{cascade_path}"
        face_cascade = cv2.CascadeClassifier(file)
        faces = face_cascade.detectMultiScale(image,
                                              scaleFactor=scale_factor,
                                              minNeighbors=min_neighbors,
                                              minSize=min_size)
        return faces
    except Exception as e:
        print(e)
        return []


def with_rectangles(image,
                    cascade_path="haarcascade_frontalface_default.xml",
                    scale_factor=1.1,
                    padding=50,
                    min_neighbors=5,
                    min_size=(30, 30)):
    faces = detect_faces(image,
                         cascade_path=cascade_path,
                         scale_factor=scale_factor,
                         min_neighbors=min_neighbors,
                         min_size=min_size)

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(image, (x - padding, y - padding),
                      (x + w + padding, y + h + padding), (255, 0, 0), 5)

    return image


def extract_faces(image, scale_factor=1.1, padding=50):
    faces = []
    for face in detect_faces(image, scale_factor=scale_factor):
        (x, y, w, h) = face
        faces.append(image[y - padding: y + h + padding,
                     x - padding: x + w + padding])
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
        "distance": result['distance'],
        "similarity_metric": result['similarity_metric'],
        "threshold": result['threshold'],
        "model": result['model']
    } for result in results if result['verified'] == True]
    return results


def verify_faces(base_image, images, models, distance_metrics):
    results = []
    for model in models:
        for distance_metric in distance_metrics:
            for image in images:
                result = verify_face(
                    base_image, image, model_name=model, distance_metric=distance_metric)
                if (result['verified'] == True):
                    return result
                else:
                    results.append(result)
    return results