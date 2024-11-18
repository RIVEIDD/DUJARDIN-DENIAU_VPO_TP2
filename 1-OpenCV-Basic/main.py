# pylint: disable=no-member

import cv2
import os
import numpy


def part_2():
    img_name: str = "peppers-256.png"
    img_folder: str = "imagesDeTest"
    src = os.path.join(os.getcwd(), img_folder, img_name)

    new_img_wh: int = 64
    img_wh: int = 256
    divisor: int = img_wh // new_img_wh

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    new_img = numpy.zeros([new_img_wh, new_img_wh, 1], dtype=numpy.uint8)

    for i in range(0, img_wh, divisor):
        for j in range(0, img_wh, divisor):
            new_img[i // divisor, j // divisor] = img[i, j]

    cv2.imshow(img_name, img)
    cv2.imshow("New " + img_name, new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part_3():
    img_name: str = "peppers-512.png"
    img_folder: str = "imagesDeTest"
    src = os.path.join(os.getcwd(), img_folder, img_name)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    dst = numpy.zeros([img.shape[0], img.shape[0], 1], dtype=numpy.uint8)
    filter = numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    cv2.filter2D(img, -1, filter, dst)

    cv2.imshow(img_name, img)
    cv2.imshow("New " + img_name, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if _name_ == "_main_":
    part_3()