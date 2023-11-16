import cv2
import numpy as np
import matplotlib.pyplot as plt
from domain.preparator import prepare
from domain.detector import extract_faces, with_rectangles, verify_faces


def plot_images(title, images):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    fig.suptitle(title)
    for i, image in enumerate(images):
        axs[i].imshow(image)
    fig.show()


def plot_faces(title, original_image, processed_image, faces):
    cols = len(faces)
    fig, axs = plt.subplots(1, cols + 2, figsize=(15, 5))
    fig.suptitle(title)
    axs[0].imshow(original_image)
    axs[1].imshow(with_rectangles(processed_image))
    for i, face in enumerate(faces):
        axs[i + 2].imshow(face)
    fig.show()


class SmartCompliance:
    """
    The main entry point to handle the KYC of a Smart Compliance process.
    Two images are required: a selfie and a photo of the ID document or passport.
    """

    def __init__(self, selfie_image, document_image) -> None:
        scale_factor = 15
        self.prepare_selfies(selfie_image, scale_factor=scale_factor)
        self.prepare_documents(document_image, scale_factor=scale_factor)
        self.selfie_documents = [self.base_image, *self.selfie_document_images]
        self.legal_documents = [self.base_image, *self.document_faces]

    def prepare_selfies(self, selfie_image, scale_factor=1):
        original_selfie, selfie = prepare(selfie_image)
        self.original_selfie_image = original_selfie
        self.selfie_image = selfie
        self.selfie_faces = extract_faces(
            self.selfie_image, scale_factor=scale_factor)
        self.base_image = max(
            self.selfie_faces, key=lambda x: x.shape[0] * x.shape[1])
        self.selfie_document_images = [
            face for face in self.selfie_faces if np.array_equal(face, self.base_image) == False]

    def prepare_documents(self, document_image, scale_factor=1):
        original_document, document = prepare(document_image)
        self.original_document_image = original_document
        self.document_image = document
        self.document_faces = extract_faces(
            self.document_image, scale_factor=scale_factor)

    def preview(self):
        plot_faces("Selfie",
                   self.original_selfie_image,
                   self.selfie_image,
                   self.selfie_faces)
        plot_faces("Document",
                   self.original_document_image,
                   self.document_image,
                   self.document_faces)
        plot_images("Selfie", self.selfie_documents)
        plot_images("Document", self.legal_documents)

    def verify(self):
        return verify_faces(self.selfie_image, self.document_image)
