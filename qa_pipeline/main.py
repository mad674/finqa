from .extractor import extract_blocks_from_pdf
from .matcher import get_relevant_blocks
from .formatter import merge_blocks
import json


def convert_pdf_to_json(pdf_path,question):
    blocks = extract_blocks_from_pdf(pdf_path)
    top_blocks = get_relevant_blocks(question, blocks, top_n=1)
    model_input = merge_blocks(top_blocks, question)
    return json.dumps(model_input, indent=2)

