import io

import streamlit as st
from PIL import Image
from tesserocr import RIL, iterate_level


def load_image():
    uploaded_file = st.file_uploader(label="Pick an image to test")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def add_margin(pil_img, top=20, right=20, bottom=20, left=20, color="black"):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def resize(im):
    try:
        # im = Image.open(img_path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= 1000:
            image = add_margin(im, 20, 20, 20, 20, "black")
            return image
        ratio = max(w, h) / 1000
        if w > h:
            w = 1000
            h = int(h / ratio)
        else:
            h = 1000
            w = int(w / ratio)
        im = im.resize((w, h))
        image = add_margin(im, 20, 20, 20, 20, "black")
        return image
    except:
        return None


def get_text(img, api):
    try:
        image = resize(img)
        level = RIL.WORD
        api.SetImage(image)
        api.Recognize()
        ri = api.GetIterator()

        words = []
        for r in iterate_level(ri, level):
            bbox = ri.BoundingBox(level)
            word = r.GetUTF8Text(level).strip()
            if word:
                words.append(word)
        return words
    except RuntimeError:
        return []


def contains(row, original):
    return any(
        w.lower() in word.lower()
        for word in row["words"]
        for w in original
        if len(w) > 1
    )
