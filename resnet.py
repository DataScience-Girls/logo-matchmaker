import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def extract_features(img_path, model):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features


def find_similar_logos(logo_path, database, model, threshold=0.7):
    logo_features = extract_features(logo_path, model)
    similar_logos = []

    for img_path, features in database.items():
        similarity = cosine_similarity(logo_features, features)
        if similarity >= threshold:
            similar_logos.append((img_path, similarity))

    return sorted(similar_logos, key=lambda x: x[1], reverse=True)


# Load pre-trained ResNet50 model (without the top layer)
resnet50_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Assume we have a dictionary called "logo_database" containing the paths of
# logo images as keys and their features as values
logo_database = {
    "data/logo1.jpg": extract_features("data/logo1.jpg", resnet50_model),
    "data/logo2.jpg": extract_features("data/logo2.jpg", resnet50_model),
    "data/logo3.jpg": extract_features("data/logo3.jpg", resnet50_model),
    "data/logo4.jpg": extract_features("data/logo4.jpg", resnet50_model),
    "data/logo5.jpg": extract_features("data/logo5.jpg", resnet50_model),
    "data/logo6.jpg": extract_features("data/logo6.jpg", resnet50_model),
    "data/logo7.jpg": extract_features("data/logo7.jpg", resnet50_model),
    "data/logo8.jpg": extract_features("data/logo8.jpg", resnet50_model),
    "data/logo9.jpg": extract_features("data/logo9.jpg", resnet50_model),
    "data/logo10.jpg": extract_features("data/logo10.jpg", resnet50_model),
    # ...
}

# Find similar logos to the target logo
target_logo_path = "data/target_logo.jpg"
similar_logos = find_similar_logos(target_logo_path, logo_database, resnet50_model)

# Print the similar logos and their similarity scores
for logo_path, similarity in similar_logos:
    print(f"Logo: {logo_path}, Similarity: {similarity[0][0]}")
