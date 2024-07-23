import os
import re

from PIL import Image
import pytesseract
from pdf2image import convert_from_path

os.environ["GOOGLE_API_KEY"] = "AIzaSyAAAR-c4ahbKgIZmRv-6zBUZWWAyJrEHqI"


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
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
        if "lợi nhuận" in txt or "doanh thu" in txt:
            documents.append(txt)

    import ipdb; ipdb.set_trace()
    return documents
