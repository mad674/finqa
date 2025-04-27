from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional


class MaskingResponse(BaseModel):
    masked_text: str

class PDFQuestionResponse(BaseModel):
    answer: str
    
class QA(BaseModel):
    question: str

class ImageMaskingResponse(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]
    masked_image: str

class ErrorResponse(BaseModel):
    error: str
class QueryIn(BaseModel):
    qa: QA
    pre_text: Optional[List[str]]  # Optional list of strings
    post_text: Optional[List[str]]  # Optional list of strings
    table: List[List[str]]  # List of lists of strings

class GenerateOut(BaseModel):
    gold_inds: List[str]
    program: str
    result: Union[float, str]
