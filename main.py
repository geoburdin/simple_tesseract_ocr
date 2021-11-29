import os
from prepare_image import process_image_for_ocr
from my_tesseract import run_tesseract, convert_pdf_to_image
from autocorrect import Speller
import cv2
import click
from glob import glob

@click.command()
@click.option('--input', help='Input Filememe. Supported extensions: pdf, jpg, png, jpeg')
@click.option('--output', default = 'result.txt', help='Filename for output: Supported extensions: txt, pdf, hocr')
@click.option('--verbose', is_flag=True, help="Print more output.")


def pipeline(input, output, verbose):
    import logging

    if verbose:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
        logging.info("Verbose output.")
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    _, tail = os.path.split(input)
    string_with_mistakes = ''
    if tail[-4:] == '.pdf':
        logging.info('Processing pdf file')
        try:
            os.mkdir('png_pages')
            logging.warning('Created /png_pages/ folder with pages in png format')
        except Exception as e:
            logging.warning('/png_pages/ folder already created')
            pass
        convert_pdf_to_image('samples/2003.00744v1_image_pdf.pdf', 'png_pages')

        for page in glob('png_pages/*.png'):
            img = process_image_for_ocr(page)
            cv2.imwrite('png_pages/{}.png'.format(page[:-4]), img)
            logging.info('Page preprocessing is completed. Result: {}'.format(page))
            run_tesseract('{}.png'.format(page[:-4]), 'output', lang='eng', extension='txt', config='--psm 3 --oem 3')
            with open('output.txt', 'r+', encoding="utf8") as file:
                string_with_mistakes += file.read()
        logging.info('Processing pdf file is completed. Post-correction is starting')
    elif tail[-4:] in ['.jpg','.png', '.jpeg']:
        img = process_image_for_ocr(input)
        cv2.imwrite('img.png', img)
        logging.info('Page preprocessing is completed. Result: img.png')
        run_tesseract('img.png', 'output', lang='eng', extension='txt', config='--psm 3 --oem 3')

        with open('output.txt', 'r+', encoding="utf8") as file:
            string_with_mistakes += '/n'+file.read()
        logging.info('Processing input file is completed. Post-correction is starting')
    else:
        logging.error('Input file format is not supported. Supported extensions: pdf, jpg, png, jpeg')

    with open(output, 'w+', encoding="utf8") as file:
        spell = Speller()
        string_without_mistakes = spell(string_with_mistakes)
        file.write(string_without_mistakes)
        logging.info('Post-proccesing is completed:')
        logging.info(string_without_mistakes)
        logging.warning('OCR is completed')


if __name__ == '__main__':
    pipeline()
