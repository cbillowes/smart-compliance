import traceback
import sys
from smart_compliance.kyc.preparator import process
from smart_compliance.kyc.detector import extract_faces_for_selfie, extract_faces_for_document
from smart_compliance.kyc.extractor import extract_text


class KycSelfie:
    def __init__(self, image) -> None:
        try:
            self.original, self.processed = process(image)
            faces, image_with_rectangles = extract_faces_for_selfie(
                self.processed)
            self.detected_faces = image_with_rectangles
            self.error = None

            if len(faces) == 0:
                self.error = "No faces detected. Please try a different image."
            else:
                self.base_image = faces[0]
                self.face = faces[1] if len(faces) > 1 else None
        except Exception as e:
            print("Could not process selfie.")
            print(e)
            print(traceback.format_exc())
            print(sys.exc_info()[2])
            self.error = "The image is not valid. Please try a different image."


class KycDocument:
    def __init__(self, image) -> None:
        try:
            self.original, self.processed = process(image)
            faces, image_with_rectangles = extract_faces_for_document(
                self.processed)
            self.detected_faces = image_with_rectangles
            self.error = None
            if len(faces) == 0:
                self.error = "No faces detected. Please try a different image."
            else:
                self.base_image = faces[0]
        except Exception as e:
            print("Could not process document: " + str(e))
            self.error = "The image is not valid. Please try a different image."

    def extract_text(self, personal_details):
        return extract_text(self.original, personal_details)


class Kyc:
    def __init__(self) -> None:
        self.selfie = None
        self.document = None

    def register_selfie(self, selfie):
        self.selfie = selfie

    def register_document(self, document):
        self.document = document
