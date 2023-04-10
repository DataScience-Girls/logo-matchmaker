import logging

import streamlit as st
from transformers import AutoFeatureExtractor, AutoModel

model_path = st.secrets["model"]

logger = logging.getLogger(__name__)


@st.cache(allow_output_mutation=True)
def get_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    model_dict = {
        "extractor": AutoFeatureExtractor.from_pretrained(model_name),
        "model": model,
        "hidden_dim": model.config.hidden_size,
    }
    return model_dict


MODELS = get_model(model_path)
