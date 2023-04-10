import streamlit as st
from scripts.embed_data import EMBED_DATASET, OCR_DATASET, fetch_similar
from scripts.models import MODELS
from scripts.utils import contains, get_text, load_image
from tesserocr import PyTessBaseAPI

test_label = 0
api = PyTessBaseAPI()


def main():
    st.title("Upload Image")
    test_sample = load_image()
    if test_sample:
        sim_ids, sim_labels = fetch_similar(
            test_sample,
            MODELS["model"],
            EMBED_DATASET["all_candidate_embeddings"],
            EMBED_DATASET["candidate_ids"],
        )
        st.header(f"Top 5 candidate similar images")

        images = [EMBED_DATASET["candidate_subset_emb"][id]["image"] for id in sim_ids]
        st.image(
            images,
            width=200,
            caption=[f"Similar image # {i+1}" for i in range(len(images))],
        )
        # for i, img in enumerate(images):
        #     st.image(img, caption=f"Similar image # {i+1}")
        ocr_target = get_text(test_sample, api)
        if ocr_target:
            OCR_DATASET["ocr"] = OCR_DATASET.apply(
                lambda row: contains(row, ocr_target), axis=1
            )
            indexes = list(OCR_DATASET[OCR_DATASET["ocr"] == True].index)
            indexes = list(set(indexes).difference(set(sim_ids)))
            images_ocr = [
                EMBED_DATASET["candidate_subset_emb"][id]["image"] for id in indexes
            ]
            if indexes:
                st.header(f"Possible similar images")
                st.image(
                    images_ocr,
                    width=200,
                    caption=[
                        f"Similar image # {i + 1}" for i in range(len(images_ocr))
                    ],
                )


if __name__ == "__main__":
    main()
