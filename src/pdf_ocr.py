import os
import re

from PIL import Image
import pytesseract
from pdf2image import convert_from_path

os.environ["GOOGLE_API_KEY"] = "AIzaSyAAAR-c4ahbKgIZmRv-6zBUZWWAyJrEHqI"


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def image_to_text(img_dir):
    extracted_text = pytesseract.image_to_string(Image.open(img_dir), config='--psm 6', lang='vie')
    return extracted_text


def imgs_to_text(dir_path):
    documents = []
    for img_path in sorted_alphanumeric(os.listdir(dir_path)):
        print(img_path)
        path = os.path.join(dir_path, img_path)
        txt = image_to_text(path)
        documents.append(txt)

    return documents


def pdf_to_image(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("the path is: ", pdf_path)
    images = convert_from_path(pdf_path)
    pdf_file_name = os.path.basename(pdf_path)[:-4]

    if not os.path.exists(os.path.join(output_dir, pdf_file_name)):
        os.mkdir(os.path.join(output_dir, pdf_file_name))

    for i in range(len(images)):
        # Save pages as images in the pdf
        page_name = 'page' + str(i) + '.jpg'
        if not os.path.exists(os.path.join(output_dir, pdf_file_name, page_name)):
            images[i].save(os.path.join(output_dir, pdf_file_name, page_name),
                           'JPEG')


if __name__ == "__main__":
    pdf_to_image("/Users/bachle/Main/Code/Projects/fin_state_read/quy2_fpt_bctc.pdf", "img_folder")
