import numpy as np
from smart_compliance.kyc.preparator import process
from smart_compliance.kyc.detector import extract_faces, with_rectangles


class KycPhoto:
    def __init__(self, image, scale_factor=1.1, padding=50) -> None:
        self.original, self.processed = process(
            image, scale_factor=scale_factor)
        faces = extract_faces(
            self.processed, scale_factor=scale_factor, padding=padding)
        self.detected_faces = with_rectangles(
            self.processed, scale_factor=scale_factor, padding=padding)

        self.base_image = None
        self.faces = []
        if len(faces) > 0:
            self.base_image = max(
                faces, key=lambda face: face.shape[0] * face.shape[1])
            self.faces = [face for face in faces if np.array_equal(
                face, self.base_image) == False]


class Kyc:
    def __init__(self) -> None:
        self.base_image = None
        self.selfie = None
        self.document = None

    def register_selfie(self, selfie):
        self.selfie = selfie
        self.base_image = selfie.base_image

    def register_document(self, document):
        self.document = document
