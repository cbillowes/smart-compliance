import cv2
from smart_compliance.kyc.detector import detect_faces

def rotate(image, angle):
    angle = angle if angle > 0 else 1
    rotation = 1  # avoid an infinite loop
    while (len(detect_faces(image)) == 0 and rotation < 360):
        try:
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            rotation += angle
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        except Exception as e:
            print("Could not rotate image: " + str(e))
            break
    return image


