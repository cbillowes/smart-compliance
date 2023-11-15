import cv2
import numpy as np
import matplotlib.pyplot as plt
from domain.preparator import prepare
from domain.detector import detect_faces, preview_faces, draw_rectangles, verify_faces


class SmartCompliance:
    """
    The main entry point to handle the KYC of a Smart Compliance process.
    Two images are required: a selfie and a photo of the ID document or passport.
    """
    def __init__(self, selfie_image, document_image) -> None:
        selfie = prepare(selfie_image)
        document = prepare(document_image)

        self.selfie_image = selfie
        self.document_image = document

    def preview(self):
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(self.selfie_image)
        axs[1].imshow(self.document_image)
        plt.show()

    def preview_selfie(self):
        preview_faces(self.selfie_image)

    def preview_document(self):
        preview_faces(self.document_image)

    def preview_rectangles(self):
        draw_rectangles(self.selfie_image)
        draw_rectangles(self.document_image)

    def verify(self):
        return verify_faces(self.selfie_image, self.document_image)