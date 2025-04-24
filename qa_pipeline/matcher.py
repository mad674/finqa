from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_blocks(question, blocks, top_n=2, threshold=0.3):
    # Encode the question into a tensor
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Create a list of concatenated block texts
    block_texts = [
        " ".join(block["pretext"] + block["table_sentences"] + block["posttext"])
        for block in blocks
    ]
    
    # If no blocks are found, return an empty list to avoid the error
    if len(block_texts) == 0:
        return []
    
    # Encode block texts into tensors
    block_embeddings = model.encode(block_texts, convert_to_tensor=True)
    
    # Compute cosine similarities between the question and each block embedding
    similarities = util.cos_sim(question_embedding, block_embeddings)[0]
    
    # Check if the maximum similarity is below the defined threshold.
    if similarities.numel() == 0 or similarities.max().item() < threshold:
        return "Insufficient data"
    
    # Get indices of the top_n most similar blocks
    top_indices = similarities.topk(k=min(top_n, len(block_texts))).indices.tolist()
    
    # Return the corresponding blocks
    return [blocks[i] for i in top_indices]
