import streamlit as st
from PIL import Image

from utils import (
    MODEL,
    check_create_dataset,
    find_similar_logos,
    load_image,
    load_logo_dataset,
)

check_create_dataset(
    st.secrets["logodatabase_file_path"], st.secrets["data_folder"], MODEL
)
LOGO_DATABASE = load_logo_dataset(st.secrets["logodatabase_file_path"])


def main():
    st.header("Logo trademarks similarity search")
    st.title("Upload Image")
    test_sample = load_image()
    if test_sample:
        print("Finding similar logo to test sample")
        similar_logos = find_similar_logos(
            test_sample.convert("RGB"), LOGO_DATABASE, MODEL, threshold=0.8
        )
        st.header("Top 5 candidate similar images")
        for logo_path, similarity in similar_logos:
            image = Image.open(logo_path)
            st.image(
                image,
                caption=f"Logo: {logo_path}, Similarity: {similarity[0][0]}",
                width=224,
            )


if __name__ == "__main__":
    main()
