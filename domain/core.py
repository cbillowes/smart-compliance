import numpy as np
import matplotlib.pyplot as plt
from domain.preparator import prepare
from domain.detector import extract_faces, with_rectangles, verify_faces

models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "DeepFace",
    "ArcFace",
]

distance_metrics = [
    "cosine",
    "euclidean",
    "euclidean_l2",
    "cosine",
]


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
        try:
            self.prepare_selfies(selfie_image, scale_factor=scale_factor)
            self.prepare_documents(document_image, scale_factor=scale_factor)
        except Exception as e:
            print(e)

    def prepare_selfies(self, selfie_image, scale_factor=1):
        original, selfie = prepare(selfie_image)
        self.original_selfie_image = original
        self.selfie_image = selfie
        self.selfie_faces = extract_faces(
            self.selfie_image, scale_factor=scale_factor)
        self.base_image = max(
            self.selfie_faces, key=lambda x: x.shape[0] * x.shape[1])
        self.selfie_document_faces = [
            face for face in self.selfie_faces if np.array_equal(face, self.base_image) == False]

    def prepare_documents(self, document_image, scale_factor=1):
        original, legal_document = prepare(document_image)
        self.original_legal_document_image = original
        self.legal_document_image = legal_document
        self.legal_document_faces = extract_faces(
            self.legal_document_image, scale_factor=scale_factor)

    def preview(self):
        plot_faces("Selfie",
                   self.original_selfie_image,
                   self.selfie_image,
                   self.selfie_faces)
        plot_faces("Document",
                   self.original_legal_document_image,
                   self.legal_document_image,
                   self.legal_document_faces)
        plot_images("Selfie", self.selfie_document_faces)
        plot_images("Document", self.legal_document_faces)

    def verify_selfie_faces(self):
        for model in models:
            for distance_metric in distance_metrics:
                selfie_results = verify_faces(self.base_image,
                                              self.selfie_document_faces,
                                              model_name=model,
                                              distance_metric=distance_metric)
                if (selfie_results['verified'] == True):
                    return selfie_results

    def verify_legal_document_faces(self):
        for model in models:
            for distance_metric in distance_metrics:
                legal_results = verify_faces(self.base_image,
                                             self.legal_document_faces,
                                             model_name=model,
                                             distance_metric=distance_metric)
                if (legal_results['verified'] == True):
                    return legal_results

    def verify(self):
        selfie_results = self.verify_selfie_faces()
        legal_results = self.verify_legal_document_faces()
        return (selfie_results, legal_results)