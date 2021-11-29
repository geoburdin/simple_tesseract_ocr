import tempfile

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


def process_image_for_ocr(file_path):
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    im_new = remove_underline(im_new)
    return im_new


def remove_underline(img):
    img = cv2.bitwise_not(img)
    kernel_clean = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(img, kernel_clean, iterations=1)
    kernel_line = np.ones((1, 80), np.uint8)
    clean_lines = cv2.erode(img, kernel_line, iterations=1)
    clean_lines = cv2.dilate(clean_lines, kernel_line, iterations=6)
    cleaned_img_without_lines = cleaned - clean_lines
    cleaned_img_without_lines = cv2.bitwise_not(cleaned_img_without_lines)

    return cleaned_img_without_lines


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(600, 600))
    return temp_filename


def image_smoothening(img):
    _, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(th1, (3, 3), 0)
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 45, 3)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image
