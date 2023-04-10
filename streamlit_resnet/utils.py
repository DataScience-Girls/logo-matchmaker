import io
import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.efficientnet_v2 import (EfficientNetV2L,
                                                           preprocess_input)
from tensorflow.keras.preprocessing import image


@st.cache_resource
def load_model():
    return EfficientNetV2L(weights="imagenet", include_top=False, pooling="avg")


def load_image():
    uploaded_file = st.file_uploader(label="Pick an image to test")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_and_preprocess_image(img, target_size=(224, 224)):
    if type(img) == str:
        img = image.load_img(img)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def extract_features(img, model):
    img_array = load_and_preprocess_image(img)
    features = model.predict(img_array)
    return features


def find_similar_logos(img, database, model, threshold=0.9):
    logo_features = extract_features(img, model)
    similar_logos = []

    for img_path, features in database.items():
        similarity = cosine_similarity(logo_features, features)
        if similarity >= threshold:
            similar_logos.append((img_path, similarity))

    return sorted(similar_logos, key=lambda x: x[1], reverse=True)[:5]


@st.cache_resource
def check_create_dataset(logodatabase_file_path, data_folder, _model):
    if os.path.exists(logodatabase_file_path):
        print(f"The file '{logodatabase_file_path}' exists.")
    else:
        print(f"The file '{logodatabase_file_path}' does not exist.")
        logo_database = {}
        for file_name in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file_name)

            # Check if the file is an image (assuming only JPEG and PNG formats)
            if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                logo_database[file_path] = extract_features(file_path, _model)

        with open(logodatabase_file_path, "wb") as file:
            pickle.dump(logo_database, file)
        print("Logo database saved to logo_database.pkl")


@st.cache_data
def load_logo_dataset(logodatabase_file_path):
    with open(logodatabase_file_path, "rb") as file:
        loaded_logo_database = pickle.load(file)
    print("Logo database loaded from logo_database.pkl")
    return loaded_logo_database


MODEL = load_model()
