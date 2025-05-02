from fastapi import FastAPI,Request, Response,APIRouter,UploadFile, Depends, HTTPException, status, File, Form
from fastapi.responses import JSONResponse, FileResponse
from schemas import QueryIn, GenerateOut, PDFQuestionResponse, MaskingResponse, ImageMaskingResponse, ErrorResponse
from retriever import generate_predicted_gold_inds
from evaluator import evaluate_program
from generator import infer, build_vocab, PointerProgramGenerator
from masking import predict_and_mask, run_final_pattern_check, BERTForNER, entity_mapping,mask_predictions
from model_retriever import BertRetriever
from qa_pipeline.main import convert_pdf_to_json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict#, List, Any, Union, Optional
import tempfile
from utils import find_most_relevant_sample, load_dataset, vectorize ,download_and_load_models,render_text_to_image, extract_text_from_file,transform
# import pdfplumber
import torch
from transformers import BertTokenizerFast, PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, DetrForObjectDetection, DetrImageProcessor, DetrConfig
import numpy as np
import json
import os
import io
import onnxruntime as ort
# import pytesseract
from PIL import Image#, ImageDraw, ImageFont
import shutil
# from docx import Document
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from contextlib import asynccontextmanager
from safetensors.torch import safe_open
import base64
import timm
os.environ["WANDB_DISABLED"] = "true"



bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
retriver_path,generator_path,vocab_path,train_path,test_path,detr_processor_path,detr_model_path,doc_summary_path,vit_model_path,ner_model_path = download_and_load_models()
MODEL_PATH = ner_model_path
MODEL_NAME = "bert-base-cased"
# Load model and tokenizer
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_model, ner_tokenizer
    # logger.info("Initializing model and tokenizer...")
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"Using device: {device}")
        
        # Load tokenizer
        ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Initialize model
        ner_model = BERTForNER(MODEL_NAME, len(entity_mapping))
        
        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            # Load model weights
            ner_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            # logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}, initializing with default weights")
        
        ner_model.to(device)
        ner_model.eval()
        yield
        # logger.info("Model and tokenizer initialized successfully")
    except Exception as e:
        # logger.error(f"Error initializing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your client app to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace the image model loading code
IMAGE_MODEL_DIR = detr_model_path
PROCESSOR_DIR = detr_processor_path
try:
    # Load model and processor
    img_config = DetrConfig.from_json_file(os.path.join(IMAGE_MODEL_DIR,"config.json"))
    img_model = DetrForObjectDetection(img_config)
    
    # Fix the device parameter in safe_open
    with safe_open(os.path.join(IMAGE_MODEL_DIR, "model.safetensors"), framework="pt") as f:
        img_state_dict = {k: f.get_tensor(k) for k in f.keys()}
    # Move model to device after loading state dict
    img_model.load_state_dict(img_state_dict)
    img_model = img_model.to(device)
    img_model.eval()
    
    if os.path.exists(os.path.join(PROCESSOR_DIR, "preprocessor_config.json")):
        img_processor = DetrImageProcessor.from_pretrained(
            PROCESSOR_DIR,
            local_files_only=True
        )
    else:
        raise FileNotFoundError(
            f"Processor files not found in {PROCESSOR_DIR}. "
            "Please run download_processor.py first."
        )
except Exception as e:
    print(f"Error loading DETR model: {str(e)}")
    raise RuntimeError("Failed to load DETR model. Please ensure all dependencies are installed.")
# Load tokenizer
SELECTED_CLASSES = ["budget", "form", "invoice"]
# Load ONNX model session
onnx_model_path = vit_model_path
MODEL_DIR = doc_summary_path 
ort_session = ort.InferenceSession(onnx_model_path)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

dataset,test_dataset =load_dataset(train_path=train_path, test_path=test_path)
# Load TF-IDF vectorizer (assuming it's already fitted)
vectorizer = vectorize(dataset,max_features=5000)  # Replace with your vectorizer if different
vocab_size = bert_tokenizer.vocab_size
retriever_model = BertRetriever(vocab_size=vocab_size,embed_size=60, num_layers=4, num_heads=12, hidden_dim=240, num_labels=2)   
retriever_model.load_state_dict(torch.load(retriver_path, map_location=device))
retriever_model.eval()

# Load Generator model
with open(vocab_path, "r") as f:
    vocab_dict = json.load(f)
    vocab = list(vocab_dict.keys())

generator_model = PointerProgramGenerator(vocab_dict)
generator_model.load_state_dict(torch.load(generator_path, map_location=device))
generator_model.eval()


#check health
@app.get("/health")
async def health_check():
    return {"status": "ok"}
##check if the server is running
@app.get("/")
async def root():
    return {"message": "fin_gpt server is running"} 


#input is only question
@app.post("/question")
async def find_relevant(request: Request):
    data = await request.json()
    user_question = data.get('question')
    
    if user_question:
        # Find the most relevant sample
        selected_sample = find_most_relevant_sample(user_question, dataset, vectorizer)

        # Extract the relevant question and any other desired info
        response_data = {
            "qa": {
                "question": selected_sample["qa"]["question"],
            },
            "pre_text": selected_sample["pre_text"],
            "post_text": selected_sample["post_text"],
            "table": selected_sample["table"],
        }
        
        # Pass the dictionary directly to run_pipeline
        return await run_pipeline(QueryIn(**response_data))
    else:
        return {"error": "No question provided"}, 400

#input is question and pdf file

@app.post("/ask_pdf", response_model=GenerateOut)
async def ask_pdf(question: str = Form(...),pdf: UploadFile = File(...)):
    # Step 1: Save the uploaded PDF to a temporary file
    pdf_path = f"temp_{pdf.filename}"
    try:
        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())

        # Step 2: Convert the PDF to JSON
        model_input = convert_pdf_to_json(pdf_path, question)
        print("model_input: ", model_input)
        # Step 3: Parse the JSON to extract fields for run_pipeline
        model_input_data = json.loads(model_input)
        query_in = QueryIn(
            qa={"question": question},
            pre_text=model_input_data.get("pretext", []),
            post_text=model_input_data.get("posttext", []),
            table=model_input_data.get("table", [])
        )

        # Step 4: Call run_pipeline with the extracted data
        result = await run_pipeline(query_in)

        # Step 5: Return the result
        return result

    except Exception as e:
        # Log the error and raise an HTTPException
        print(f"Error in /ask_pdf: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.")

    finally:
        # Step 6: Clean up the temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

#input is question and pretext,posttext,table
@app.post("/retrive", response_model=GenerateOut)
async def run_pipeline(data: QueryIn):
    # Step 1: Format input record for retriever
      # Parse the JSON body
    # print("data: ", data)
    record = {
        "qa": {"question": data.qa.question},
        "pre_text": data.pre_text,
        "post_text": data.post_text,
        "table": data.table,
    }
    # record=data
    # Step 2: Use retriever to get gold_inds
    gold_inds_raw = generate_predicted_gold_inds(record, retriever_model, bert_tokenizer,threshold=0,num_candidates=2)
    # print("gold_inds_raw: ", gold_inds_raw)
    gold_inds = {k: v["sentence"] for k, v in gold_inds_raw}
    # return {"gold_inds": gold_inds}
    # Step 3: Use generator to get program
    full_input = data.qa.question + " " + " ".join([v for v in gold_inds.values() if any(c.isdigit() for c in v)])
    print(full_input)
    encoded = bert_tokenizer(full_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512, return_offsets_mapping=True)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))
    sample = {
        "input_ids": encoded["input_ids"].squeeze(0),
        "input_mask": encoded["attention_mask"].squeeze(0),
        "input_tokens": input_tokens
    }
    program = infer(generator_model, sample, vocab_dict)
    # Step 4: Evaluate the program
    program=" , ".join(program[:-1])
    result = evaluate_program(program, data.table)
    gold_inds=list(gold_inds.values())
    # Step 5: Return all
    return GenerateOut(gold_inds=gold_inds, program=program, result=str(result))



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1].lower()

    if extension not in ["jpg", "jpeg", "png", "bmp", "txt", "docx", "pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        contents = await file.read()

        # Convert to image depending on file type
        if extension in ["jpg", "jpeg", "png", "bmp"]:
            image = Image.open(io.BytesIO(contents)).convert("RGB")

        elif extension == "txt":
            text = contents.decode("utf-8")
            image = render_text_to_image(text)

        elif extension == "docx":
            text = extract_text_from_file(file,contents)
            image = render_text_to_image(text)

        elif extension == "pdf":
            text = extract_text_from_file(file,contents)
            image = render_text_to_image(text)

        # Predict
        image_tensor = transform(image).unsqueeze(0).numpy()
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: image_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        logits = ort_outs[0]
        probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy().flatten()

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return JSONResponse(content={
            "prediction": SELECTED_CLASSES[predicted_class],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()

        # ✅ Check allowed extensions
        allowed_extensions = (".pdf", ".docx", ".txt", ".jpeg", ".jpg", ".png")
        if not filename.endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT, JPG, JPEG, PNG files are allowed.")

        # ✅ Read file contents
        content = await file.read()

        # ✅ Extract text
        text = extract_text_from_file(file, content)
        if not text.strip():
            return {"error": "No readable text found in the file."}

        # ✅ Tokenize input and generate summary using Pegasus
        encoded = pegasus_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to(device)

        with torch.no_grad():
            summary_ids = pegasus_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# File upload and masking endpoint
@app.post("/mask-file", response_model=MaskingResponse)
async def mask_file(file: UploadFile = File(...)):
    """
    Upload a file and mask sensitive information in it.
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Read the file content
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Process the content
        original_content = content
        masked_content = predict_and_mask(ner_tokenizer,ner_model,content,device)
        
        # Run a final check to ensure correct masking
        masked_content = run_final_pattern_check(masked_content, original_content)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Return the masked content
        return MaskingResponse(masked_text=masked_content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Text masking endpoint
@app.post("/mask-text", response_model=MaskingResponse)
async def mask_text(request: Dict[str, str]):
    """
    Mask sensitive information in provided text.
    """
    text = request.get("text", "")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        original_text = text
        masked_text = predict_and_mask(ner_tokenizer,ner_model,text,device)
        
        # Run a final check to ensure correct masking
        masked_text = run_final_pattern_check(masked_text, original_text)
        
        return MaskingResponse(masked_text=masked_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post("/mask-image", response_model=ImageMaskingResponse)
async def mask_image(file: UploadFile = File(...)):
    """
    Mask sensitive information in an image.
    """
    # Validate file type
    if not file.filename:
        print("Error: No file provided")
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Infer file type from extension if Content-Type is application/octet-stream
    allowed_types = {"image/jpeg", "image/png", "image/bmp"}
    file_extension = file.filename.split(".")[-1].lower()
    inferred_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp"
    }.get(file_extension)

    if file.content_type == "application/octet-stream" and inferred_type not in allowed_types:
        print(f"Error: Unsupported file type inferred from extension: {file_extension}")
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Must be one of: {', '.join(allowed_types)}"
        )
    if file.content_type not in allowed_types and inferred_type not in allowed_types:
        print(f"Error: Unsupported file type {file.content_type} or inferred type {inferred_type}")
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Must be one of: {', '.join(allowed_types)}"
        )

    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            print("Error: Empty file")
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Open and convert image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            print(f"Error: Invalid image file - {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Process with model
        try:
            inputs = img_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # print(inputs)
            with torch.no_grad():
                outputs = img_model(**inputs)
            # Postprocess outputs
            target_sizes = torch.tensor([image.size[::-1]])
            results = img_processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.9
            )[0]

            # Check if any objects were detected
            if len(results["boxes"]) == 0:
                print("No objects detected in the image")
                return ImageMaskingResponse(
                    boxes=[],
                    labels=[],
                    scores=[],
                    masked_image=""
                )

            # Generate masked image
            masked_image = mask_predictions(image, results["boxes"])
            
            # Convert to base64
            buffered = io.BytesIO()
            masked_image.save(buffered, format="PNG")
            masked_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Prepare response
            return ImageMaskingResponse(
                boxes=results["boxes"].tolist(),
                labels=[
                    img_model.config.id2label.get(label.item(), f"Class {label.item()}")
                    for label in results["labels"]
                ],
                scores=results["scores"].tolist(),
                masked_image=f"data:image/png;base64,{masked_image_base64}"
            )
            
        except Exception as e:
            print(f"Model inference error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing image with model"
            )

    except HTTPException as http_exc:
        print(f"HTTP Exception: {http_exc.detail}")
        raise
    except Exception as e:
        print(f"Unexpected error in mask_image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the image"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)