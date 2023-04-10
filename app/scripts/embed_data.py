import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from datasets import load_dataset
from scripts.models import MODELS
from scripts.utils import get_text
from tesserocr import PyTessBaseAPI
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
api = PyTessBaseAPI()
path = st.secrets["path"]
seed = 42
num_samples = 100
dataset = load_dataset("imagefolder", data_dir=path, drop_labels=False)
candidate_subset = dataset["train"]  # .shuffle(seed=seed).select(range(num_samples))

transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * MODELS["extractor"].size["height"])),
        T.CenterCrop(MODELS["extractor"].size["height"]),
        T.ToTensor(),
        T.Normalize(
            mean=MODELS["extractor"].image_mean, std=MODELS["extractor"].image_std
        ),
    ]
)


def make_cand_id(subset):
    candidate_ids = []
    for id in tqdm(range(len(subset))):
        label = subset[id]["label"]

        entry = str(id) + "_" + str(label)

        candidate_ids.append(entry)
    return candidate_ids


@st.cache(allow_output_mutation=True)
def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image.convert("RGB")) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def embed_dataset(dataset, model):
    extract_fn = extract_embeddings(model.to(device))
    candidate_subset_emb = dataset.map(extract_fn, batched=True, batch_size=24)
    candidate_ids = make_cand_id(candidate_subset_emb)
    all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
    all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)
    return {
        "candidate_subset_emb": candidate_subset_emb,
        "candidate_ids": candidate_ids,
        "all_candidate_embeddings": all_candidate_embeddings,
    }


def get_ocr_dataset(dataset, api):
    df_dict = {"words": [get_text(image["image"], api) for image in tqdm(dataset)]}
    return pd.DataFrame.from_dict(df_dict)


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def fetch_similar(image, model, all_candidate_embeddings, candidate_ids, top_k=5):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    image_transformed = transformation_chain(image.convert("RGB")).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    # Comute the embedding.
    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    sim_scores = compute_scores(all_candidate_embeddings, query_embeddings)
    similarity_mapping = dict(zip(candidate_ids, sim_scores))

    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id_entries = list(similarity_mapping_sorted.keys())[:top_k]

    ids = list(map(lambda x: int(x.split("_")[0]), id_entries))
    labels = id_entries
    return ids, labels


EMBED_DATASET = embed_dataset(candidate_subset, MODELS["model"])
OCR_DATASET = get_ocr_dataset(candidate_subset, api)
