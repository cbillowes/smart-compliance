import cv2


def rotate(image, loop_while, angle):
    angle = angle if angle > 0 else 1
    rotation = 1 # avoid an infinite loop
    while (loop_while(image) == True and rotation < 360):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        rotation += angle
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    return image
