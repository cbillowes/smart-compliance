from deepface import DeepFace
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import cv2


class SmartComplianceDocument():
    def __init__(self, image,
                 cascade_path="haarcascade_frontalface_default.xml",
                 scaleFactor=1.1,
                 minNeighbors=5,
                 minSize=(30, 30)):
        self.image = image
        self.scale_factor = scaleFactor
        self.min_neighbors = minNeighbors
        self.min_size = minSize

        file = f"{cv2.data.haarcascades}{cascade_path}"
        self.face_cascade = cv2.CascadeClassifier(file)

    def preview(self):
        plt.imshow(self.image)
        plt.show()

    def preview_faces(self):
        faces = self.detect()
        for i, face in enumerate(faces):
            (x, y, w, h) = face
            plt.imshow(self.image[y:y + h, x:x + w])
            plt.show()

    def detect(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray,
                                                  scaleFactor=self.scale_factor,
                                                  minNeighbors=self.min_neighbors,
                                                  minSize=self.min_size)

    def has_detected_at_least_one_face(self):
        return len(self.detect()) > 0

    def rotate(self, adjust_by=45, scale=1.0):
        adjusted_angle = 0
        while self.has_detected_at_least_one_face() == False and adjusted_angle < 360:
            adjusted_angle += adjust_by
            (h, w) = self.image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, adjusted_angle, scale)
            self.image = cv2.warpAffine(self.image, M, (w, h))
        return self.has_detected_at_least_one_face()

    def denoise(self):
        # TODO
        return

    def crop(self):
        faces = self.detect()
        images = []
        for i, face in enumerate(faces):
            (x, y, w, h) = face
            images.append(self.image[y:y + h, x:x + w])
        self.faces = images

    def extract_characters(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extracted_text = pytesseract.image_to_string(thresh, lang="eng")
        return extracted_text.strip().replace("\n", " ")


class SmartCompliance():
    def __init__(self, selfie, id_document) -> None:
        self.selfie = selfie
        self.id_document = id_document

    def verify(self):
        self.selfie.crop()
        self.id_document.crop()
        faces = [self.selfie.faces, self.id_document.faces]
        return

        # find the correct sized images for base and sample
