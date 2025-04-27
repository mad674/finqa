import numpy as np
import os
import gdown
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision import transforms
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image, ImageDraw, ImageFont
import docx
import io 
from transformers import DetrImageProcessor
import pdfplumber
import easyocr
# from pytesseract import image_to_string
from fastapi.responses import JSONResponse
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Define the same preprocessing pipeline used during training
reader = easyocr.Reader(['en'],gpu=torch.cuda.is_available()) 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
def render_text_to_image(text, width=800, height=1000):
    """Render text to image."""
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    lines = text.split("\n")
    y = 10
    for line in lines:
        if y > height - 30:
            break  # Prevent overflow
        draw.text((10, y), line[:120], fill="black", font=font)  # Limit line width
        y += 25
    return image

def extract_text_from_file(file: UploadFile, content: bytes) -> str:
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return content.decode("utf-8")

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(content)).convert("RGB")
        # return pytesseract.image_to_string(image)
        # return image_to_string(image) # Specify the language(s)
        result = reader.readtext(np.array(image), detail=0)  # `detail=0` returns only the text
        return "\n".join(result)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: pdf, docx, txt, jpg, jpeg, png")

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
            "file_id": "1JA2pgCbPTqqjOtur6wG0p_zgiiI8SCRS",  # Replace with your Google Drive file ID
            "output_path": "models/120aBRS.pt"
        },
        "generator_model": {
            "file_id": "1M0YWjjOfpJ5pEZNxTQEiRYFIvkZu7v6S",  # Replace with your Google Drive file ID
            "output_path": "models/100egen.pt"
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
        },
        "detr_processor": {
            "file_id": "1XEePJlhpk_smKWfwE3ZJLnlyOiE553vz",  # Replace with your Google Drive file ID
            "output_path": "models/detr_processor"
        },
        "detr_model": {
            "file_id": "1vyFvWoImwhW9ApGAUMtSyHLT7dDDxKOA",  # Replace with your Google Drive file ID
            "output_path": "models/detr_model"
        },
        "doc_summary":{
            "file_id": "19QqLpZymJetTdG2k04JDyXWfoJNeGMGu",  # Replace with your Google Drive file ID
            "output_path": "models/doc_summary_model"
        },
        "vit_model":{
            "file_id": "1SoHRU9vLQeQjVzMm3D5o4GzXdZjdXEqd",  # Replace with your Google Drive file ID
            "output_path": "models/vit_model.onnx"
        },
        "ner_model":{
            "file_id": "1ai_DqcnqoPyyxQkF5VWTwmsRt22y5mIL",  # Replace with your Google Drive file ID
            "output_path": "models/best_ner_model.pt"
        },

    }

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Download all required files
    for model_name, model_info in model_files.items():
        download_model(model_info["file_id"], model_info["output_path"])
    
    print("All models and data files downloaded successfully.")
    
    return model_files["retriever_model"]["output_path"], model_files["generator_model"]["output_path"], model_files["vocab_file"]["output_path"], model_files["train_data"]["output_path"], model_files["test_data"]["output_path"], model_files["detr_processor"]["output_path"], model_files["detr_model"]["output_path"], model_files["doc_summary"]["output_path"], model_files["vit_model"]["output_path"], model_files["ner_model"]["output_path"]