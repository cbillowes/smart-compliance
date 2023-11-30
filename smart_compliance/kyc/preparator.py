import cv2
from smart_compliance.kyc.rotate import rotate


def process(image, scale_factor=1.1):
    processed = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    # processed = rotate(processed, angle=10)
    return (image, processed)
