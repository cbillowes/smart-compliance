from deepface import DeepFace
import cv2
import pandas as pd
import matplotlib.pyplot as plt


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


def extract_faces(image, scale_factor=1.1):
    padding = 50 # px
    faces = []
    for face in detect_faces(image, scale_factor=scale_factor):
        (x, y, w, h) = face
        faces.append(image[y - padding : y + h + padding, x - padding : x + w + padding])
    return faces


def verify_faces(base_image, faces, model_name="VGG-Face", distance_metric="euclidean_l2", detector_backend="opencv"):
    results = []

    for face in faces:
        print(model_name, distance_metric)
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0] = plt.imshow(base_image)
        # axs[1] = plt.imshow(face)
        # plt.show()
        try:
            result = DeepFace.verify(img1_path=base_image,
                                    img2_path=face,
                                    model_name=model_name,
                                    distance_metric=distance_metric,
                                    detector_backend=detector_backend,
                                    enforce_detection=False)
            if (result['verified'] == True):
                plt.imshow(base_image)
                plt.show()
                plt.imshow(face)
                plt.show()
                return result
            else:
                results.append(result)
        except Exception as e:
            print(e)
            results.append({'verified': False, 'reason': 'No face detected'})

    return (pd.DataFrame(results)[['verified']] == True).to_dict('list')
