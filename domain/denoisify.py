import cv2
import numpy as np


def get_crop_mask(mask):
    return np.zeros_like(mask)


def get_mask(frame, kernel, iterations=1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canned = cv2.Canny(gray, 100, 200)
    mask = cv2.dilate(canned, kernel, iterations=iterations)
    return mask


def apply_mask(frame, mask):
    # find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find big contours
    biggest_cntr = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_cntr = contour

    # draw contours
    crop_mask = np.zeros_like(mask)
    cv2.drawContours(crop_mask, [biggest_cntr], -1, (255), -1)
    return crop_mask


def invert_mask(mask):
    inverted = cv2.bitwise_not(mask)
    # contours again
    contours, _ = cv2.findContours(
        inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find small contours
    small_cntrs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20000:
            small_cntrs.append(contour)

    # draw on mask
    crop_mask = get_crop_mask(mask)
    cv2.drawContours(crop_mask, small_cntrs, -1, (255), -1)
    return inverted


def smooth_jaggies(crop_mask, kernel):
    crop_mask = cv2.erode(crop_mask, kernel, iterations=1)
    crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)
    crop_mask = cv2.medianBlur(crop_mask, 5)
    return crop_mask


def crop(frame, crop_mask):
    crop = np.zeros_like(frame)
    crop[crop_mask == 255] = frame[crop_mask == 255]


def denoisify(frame):
    """
    Sometimes the background of an image can be too noisy.
    This function will remove the background of an object for a given frame of an image.
    https://stackoverflow.com/questions/45053513/how-do-i-detect-and-remove-blurred-background-from-image
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = get_mask(frame, kernel)
    applied_mask = apply_mask(frame, mask)
    inverted_mask = invert_mask(applied_mask)
    crop_mask = smooth_jaggies(inverted_mask, kernel)
    crop = np.zeros_like(frame);
    crop[crop_mask == 0] = frame[crop_mask == 0];
    return crop
