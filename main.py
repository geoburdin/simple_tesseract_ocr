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

    # (1) clean up noises
    kernel_clean = np.ones((2,2),np.uint8)
    cleaned = cv2.erode(img, kernel_clean, iterations=1)

    # (2) Extract lines
    kernel_line = np.ones((1, 10), np.uint8)
    clean_lines = cv2.erode(img, kernel_line, iterations=5)
    clean_lines = cv2.dilate(clean_lines, kernel_line, iterations=10)

    # (3) Subtract lines
    cleaned_img_without_lines = cleaned - clean_lines
    cleaned_img_without_lines = cv2.bitwise_not(cleaned_img_without_lines)

    return cleaned_img_without_lines

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 3)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


img = process_image_for_ocr('samples/ocr_sample.png')
cv2.imwrite('img.png', img)

import shlex
import subprocess
import sys
from pkgutil import find_loader


try:
    from PIL import Image
except ImportError:
    import Image


tesseract_cmd = 'tesseract'

numpy_installed = find_loader('numpy') is not None
if numpy_installed:
    from numpy import ndarray


def run_tesseract(input_filename, output_filename_base,
    extension, lang, config='', nice=0, timeout=0):
    cmd_args = []

    if not sys.platform.startswith('win32') and nice != 0:
        cmd_args += ('nice', '-n', str(nice))

    cmd_args += (tesseract_cmd, input_filename, output_filename_base)

    if lang is not None:
        cmd_args += ('-l', lang)

    if config:
        cmd_args += shlex.split(config)

    if extension:
        cmd_args.append(extension)

    try:
        proc = subprocess.Popen(cmd_args)
    except OSError as e:
        raise e

run_tesseract('img.png', 'output', lang='eng', extension='hocr', config = '--psm 3 --oem 3')
