import shlex
import subprocess

tesseract_cmd = 'tesseract'


def run_tesseract(input_filename, output_filename_base,
    extension, lang, config=''):
    cmd_args = []

    cmd_args += (tesseract_cmd, input_filename, output_filename_base)

    if lang is not None:
        cmd_args += ('-l', lang)

    if config:
        cmd_args += shlex.split(config)

    if extension:
        cmd_args.append(extension)

    try:
        proc = subprocess.call(cmd_args)

    except OSError as e:
        raise e


import fitz

def convert_pdf_to_image(pdf_input_path, jpeg_output_path):
    doc = fitz.open(pdf_input_path)
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(matrix = fitz.Matrix(5, 5))  # render page to an image
        pix.save(jpeg_output_path+"/page-%i.png" % page.number)