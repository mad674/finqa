import numpy as np
import os
import gdown
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
def find_most_relevant_sample(user_question, dataset, vectorizer):
    questions = [sample["qa"]["question"] for sample in dataset]
    question_vectors = vectorizer.transform(questions).toarray()

    user_vector = vectorizer.transform([user_question]).toarray()
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_match_index = np.argmax(similarities)
    return dataset[best_match_index]

def load_dataset(train_path="data/train.json", test_path="data/test.json"):
    with open(train_path, "r") as f:
        dataset = json.load(f)
    with open(test_path, "r") as f:
        test_dataset = json.load(f)

    dataset.extend(test_dataset)

    return dataset,test_dataset


def vectorize(dataset, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=5000)  # Replace with your vectorizer if different
    all_texts = [sample["qa"]["question"] for sample in dataset]
    vectorizer.fit(all_texts)
    return vectorizer

def download_model(file_id, output_path):
    """
    Downloads a file from Google Drive using gdown.
    :param file_id: The Google Drive file ID.
    :param output_path: The local path to save the file.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")

def download_and_load_models():
    model_files = {
        "retriever_model": {
            "file_id": "1axPDNotrQrRedr6EsEhrKVWrQiXkQf5y",  # Replace with your Google Drive file ID
            "output_path": "models/100aBRS.pt"
        },
        "generator_model": {
            "file_id": "1Qr6CNugiCmaFIUxwhhzSbssVuJam-IkG",  # Replace with your Google Drive file ID
            "output_path": "models/40egen.pt"
        },
        "vocab_file": {
            "file_id": "11yTbdEJvxe4IWBmaxzFPAFSfOJ-krxjj",  # Replace with your Google Drive file ID
            "output_path": "data/finqa_vocab.json"
        },
        "train_data": {
            "file_id": "1k0GgkIXCPic_EgX1m_wc0ZiXZnMYO1Dk",  # Replace with your Google Drive file ID
            "output_path": "data/train.json"
        },
        "test_data": {
            "file_id": "1KNZXY7Axu2bEDx3mXV6tUpREtQF1oMlx",  # Replace with your Google Drive file ID
            "output_path": "data/test.json"
        }
    }

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Download all required files
    for model_name, model_info in model_files.items():
        download_model(model_info["file_id"], model_info["output_path"])
    
    print("All models and data files downloaded successfully.")
    
    return model_files["retriever_model"]["output_path"], model_files["generator_model"]["output_path"], model_files["vocab_file"]["output_path"], model_files["train_data"]["output_path"], model_files["test_data"]["output_path"]