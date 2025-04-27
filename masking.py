import torch
from transformers import AutoTokenizer, AutoModel
import re
import os
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import UploadFile, HTTPException

# Entity label mappings - with specific PII types
entity_mapping = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,     # Person
    'B-ORG': 3, 'I-ORG': 4,     # Organization
    'B-LOC': 5, 'I-LOC': 6,     # Location
    'B-MISC': 7, 'I-MISC': 8,   # Miscellaneous
    'B-PHONE': 9, 'I-PHONE': 10, # Phone numbers
    'B-CARD': 11, 'I-CARD': 12, # Credit card information
    'B-EMAIL': 13, 'I-EMAIL': 14, # Email addresses
    'B-DATE': 15, 'I-DATE': 16,  # Dates
}

# Reverse mapping for decoding predictions
id2label = {v: k for k, v in entity_mapping.items()}

# Map of regex detection type to proper entity type for masking
regex_to_entity_type = {
    'PHONE': 'PHONE',
    'CARD': 'CARD',
    'EMAIL': 'EMAIL',
    'DATE': 'DATE',
    'PER': 'PER',
    'ORG': 'ORG',
    'LOC': 'LOC',
    'MISC': 'MISC'
}

# Improved and more comprehensive regex patterns for direct PII detection
patterns = {
    # Phone patterns covering international formats, Indian formats, and various separators
    'PHONE': [
        r'\+\d{1,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}',  # International format: +XX XXX XXX XXXX
        r'(\+91[\s-]?)?[0]?(91)?[6789]\d{9}',  # Indian mobile numbers
        r'\(\d{3}\)\s*\d{3}[\s-]?\d{4}',  # (123) 456-7890
        r'\d{3}[\s-]?\d{3}[\s-]?\d{4}',  # 123-456-7890
        r'\d{5}[\s-]?\d{5}',  # 12345 12345
    ],
    
    # Credit card patterns covering major formats with or without separators
    'CARD': [
        r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',  # 16-digit cards with optional separators
        r'\d{4}[\s-]?\d{6}[\s-]?\d{4}[\s-]?\d{1}',  # 15-digit cards (AMEX format)
        r'\d{4}[\s-]?\d{4}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{4}',  # Other formats
    ],
    
    # Email pattern with comprehensive TLD coverage
    'EMAIL': [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Standard email format
    ],
    
    # Date patterns covering common formats
    'DATE': [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY, MM/DD/YYYY
        r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}\b',  # DD Mon YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{2,4}\b',  # Mon DD, YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
    ]
}


# NER model class
class BERTForNER(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Track all matches to avoid conflicts
class EntityMatch:
    def __init__(self, start, end, text, entity_type):
        self.start = start
        self.end = end
        self.text = text
        self.entity_type = entity_type
    
    def overlaps(self, other):
        """Check if this match overlaps with another match"""
        return (self.start <= other.end and other.start <= self.end)
    
    def contains(self, other):
        """Check if this match fully contains another match"""
        return (self.start <= other.start and self.end >= other.end)
    
    def get_mask(self):
        """Return the appropriate mask for this entity"""
        return f"[{self.entity_type}_MASK]"

def apply_regex_masking(text):
    """
    Apply regex-based masking to directly catch obvious PII patterns
    Returns detected matches.
    """
    # Track all matches
    matches = []
    
    # Find all matches across all patterns
    for regex_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                matches.append(EntityMatch(
                    start, 
                    end, 
                    match.group(0), 
                    regex_type
                ))
    
    return matches

def tokenize_and_predict(tokenizer, model, text,device):
    """Use NER model to identify entities in text"""
    model_matches = []
    
    # Split text into sentences for better processing
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    current_pos = 0
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence:
            current_pos += len(sentence) + 1  # +1 for the space or newline
            continue
        
        # Find the exact position of this sentence in the original text
        # to ensure we get correct character positions
        sentence_start = text.find(sentence, current_pos)
        if sentence_start == -1:
            # This should not happen, but handle it just in case
            current_pos += len(sentence) + 1
            continue
        
        current_pos = sentence_start + len(sentence)
        
        # Split sentence into words
        words = sentence.split()
        word_positions = []
        
        # Track the position of each word in the original text
        word_start = sentence_start
        for word in words:
            word_pos = sentence.find(word, word_start - sentence_start)
            if word_pos != -1:
                abs_word_pos = sentence_start + word_pos
                word_positions.append((abs_word_pos, abs_word_pos + len(word)))
                word_start = abs_word_pos + len(word) - sentence_start
        
        # Skip if we can't properly align words
        if not word_positions or len(word_positions) != len(words):
            continue
        
        # Tokenize input for the model
        tokenized_input = tokenizer(
            words, 
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            output = model(
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                token_type_ids=tokenized_input["token_type_ids"]
            )
        
        # Get predicted labels
        predictions = torch.argmax(output, dim=2).cpu().numpy()[0]
        
        # Align predictions with words
        word_ids = tokenized_input.word_ids()
        
        # Group words into entities
        i = 0
        while i < len(words):
            if i >= len(word_positions):
                break
                
            # Get the NER label for this word
            token_indices = [idx for idx, wid in enumerate(word_ids) if wid == i]
            if not token_indices:
                i += 1
                continue
                
            token_labels = [id2label.get(predictions[idx], "O") for idx in token_indices]
            
            # Check if this word starts an entity
            if any(label.startswith("B-") for label in token_labels):
                # Extract entity type
                entity_type = next((label.split("-")[1] for label in token_labels if label.startswith("B-")), None)
                
                if entity_type:
                    entity_start = word_positions[i][0]
                    j = i + 1
                    
                    # Continue collecting words that are part of this entity
                    while j < len(words) and j < len(word_positions):
                        j_token_indices = [idx for idx, wid in enumerate(word_ids) if wid == j]
                        j_token_labels = [id2label.get(predictions[idx], "O") for idx in j_token_indices]
                        
                        if any(label.startswith("I-") and label.split("-")[1] == entity_type for label in j_token_labels):
                            j += 1
                        else:
                            break
                    
                    entity_end = word_positions[j-1][1]
                    entity_text = text[entity_start:entity_end]
                    
                    model_matches.append(EntityMatch(
                        entity_start,
                        entity_end,
                        entity_text,
                        entity_type
                    ))
                    
                    i = j  # Skip to the end of this entity
                    continue
            
            i += 1
    
    return model_matches

def predict_and_mask(tokenizer, model, text,device):
    """
    Comprehensive masking using both regex patterns and NER model
    """
    # Step 1: Get matches from regex patterns
    regex_matches = apply_regex_masking(text)
    
    # Step 2: Get matches from NER model
    model_matches = []
    try:
        if 'model' in globals() and model is not None:
            model_matches = tokenize_and_predict(tokenizer, model, text,device)
    except Exception as e:
        print(f"Model-based masking failed: {str(e)}. Continuing with regex-only masking.")
    
    # Combine all matches
    all_matches = regex_matches + model_matches
    
    # Step 3: Resolve overlapping matches (prioritize more specific entity types)
    priority_order = {
        'EMAIL': 1,
        'PHONE': 2,
        'CARD': 3,
        'DATE': 4,
        'PER': 5,
        'ORG': 6,
        'LOC': 7,
        'MISC': 8
    }
    
    # Sort by start position
    all_matches.sort(key=lambda m: m.start)
    
    # Filter overlapping matches
    final_matches = []
    for match in all_matches:
        # Check if this match overlaps with any existing match
        overlapping = False
        for i, existing in enumerate(final_matches):
            if match.overlaps(existing):
                # If current match contains the existing match and has higher priority,
                # replace the existing match
                if match.contains(existing) and priority_order.get(match.entity_type, 9) < priority_order.get(existing.entity_type, 9):
                    final_matches[i] = match
                    overlapping = True
                    break
                # If existing match contains the current match and has higher priority,
                # skip the current match
                elif existing.contains(match) and priority_order.get(existing.entity_type, 9) <= priority_order.get(match.entity_type, 9):
                    overlapping = True
                    break
                # If they partially overlap, keep the one with higher priority
                else:
                    if priority_order.get(match.entity_type, 9) < priority_order.get(existing.entity_type, 9):
                        final_matches[i] = match
                    overlapping = True
                    break
        
        if not overlapping:
            final_matches.append(match)
    
    # Sort by start position in reverse order to replace from end to start
    final_matches.sort(key=lambda m: m.start, reverse=True)
    
    # Apply masking
    masked_text = text
    for match in final_matches:
        masked_text = masked_text[:match.start] + match.get_mask() + masked_text[match.end:]
    
    return masked_text


def mask_predictions(image, boxes, fill_color=(0, 0, 0)):
    masked_image = image.copy()
    draw = ImageDraw.Draw(masked_image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
    return masked_image

def run_final_pattern_check(text, original_text):
    """
    Final check to ensure standard PII patterns are masked with the correct entity type
    """
    # Define strict patterns for the most common PII formats
    strict_patterns = {
        # Email check
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': 'EMAIL',
        
        # Phone number check
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}': 'PHONE',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}': 'PHONE',
        
        # Credit card check
        r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}': 'CARD',
        
        # Date check
        r'\d{1,2}/\d{1,2}/\d{2,4}': 'DATE',
        r'\d{4}-\d{1,2}-\d{1,2}': 'DATE'
    }
    
    # Check the original text for any missed patterns
    matches = []
    for pattern, entity_type in strict_patterns.items():
        for match in re.finditer(pattern, original_text):
            start, end = match.span()
            match_text = match.group(0)
            
            # Check if this pattern is already masked in the masked text
            # Find the corresponding position in the masked text
            text_before = original_text[:start]
            mask_count_before = sum(1 for _ in re.finditer(r'\[\w+_MASK\]', text[:len(text_before)]))
            
            # Adjust position to account for masks already added
            adjusted_start = start + mask_count_before * 10  # Approximate adjustment
            context_range = 20
            
            # Look in the vicinity for a mask
            context_start = max(0, adjusted_start - context_range)
            context_end = min(len(text), adjusted_start + len(match_text) + context_range)
            context = text[context_start:context_end]
            
            # If no mask is found in the context, add this match
            if not re.search(r'\[\w+_MASK\]', context):
                matches.append(EntityMatch(start, end, match_text, entity_type))
    
    # Sort matches by position in reverse order
    matches.sort(key=lambda m: m.start, reverse=True)
    
    # Apply additional masking for any missed patterns
    for match in matches:
        # Find the corresponding position in the masked text
        text_before_match = original_text[:match.start]
        text_after_match = original_text[match.end:]
        
        # Find the location in the masked text
        for i in range(len(text)):
            # Try to locate the match by surrounding context
            context_before = text_before_match[-min(20, len(text_before_match)):]
            context_after = text_after_match[:min(20, len(text_after_match))]
            
            if i > 0 and text[i-len(context_before):i].endswith(context_before) and \
               i+len(match.text) < len(text) and text[i+len(match.text):i+len(match.text)+len(context_after)].startswith(context_after) and \
               match.text in text[i:i+len(match.text)]:
                
                # Insert mask
                text = text[:i] + match.get_mask() + text[i+len(match.text):]
                break
    
    return text