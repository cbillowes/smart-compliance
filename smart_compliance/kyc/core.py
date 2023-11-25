import numpy as np
from smart_compliance.kyc.preparator import process
from smart_compliance.kyc.detector import extract_faces


class KycPhoto:
    def __init__(self, image) -> None:
        self.original, self.processed = process(image)
        faces, image_with_rectangles = extract_faces(self.processed)
        self.detected_faces = image_with_rectangles
        self.base_image = faces[0]
        self.face = faces[1]


class Kyc:
    def __init__(self) -> None:
        self.selfie = None
        self.document = None

    def register_selfie(self, selfie):
        self.selfie = selfie

    def register_document(self, document):
        self.document = document
