
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
import pprint


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
def table_to_sentences(table, base_key="table"):

      sentences = {}
      if not table or len(table) < 2:
          return sentences  # Nothing to convert

      header = table[0]
      num_cols = len(header)

      col_labels = header[1:]
      for idx, row in enumerate(table[1:], start=1):
          # Ensure row has at least one label cell and one value.
          if len(row) < 2:
              continue
          label = row[0].strip()
          values = row[1:]
          # If the number of values does not match the number of column labels, skip.
          if len(values) != len(col_labels):
              continue
          parts = []
          for col, value in zip(col_labels, values):
              # Construct phrase: "the {label} of {col} is {value}"
              parts.append(f"{header[0].strip()} the {label} of {col.strip()} is {value.strip()}")
          sentence = " ; ".join(parts)
          key = f"{base_key}_{idx}"
          sentences[key] = sentence

      return sentences


def extract_candidates(record):
    """
    Extract candidates from pre_text, post_text, and table fields,
    assigning each candidate a unique key.
    """
    candidates = {}
    # Process pre_text sentences.
    for idx, sentence in enumerate(record.get("pre_text", [])):
        candidates[f"text_{idx}"] = sentence
    # Process post_text sentences (continue numbering after pre_text).
    offset = len(candidates)
    for idx, sentence in enumerate(record.get("post_text", [])):
        candidates[f"text_{offset + idx}"] = sentence
    # Process table rows if available.
    if "table" in record:
        table_candidates = table_to_sentences(record["table"], base_key="table")
        candidates.update(table_candidates)
    return candidates


def generate_predicted_gold_inds(record, model, tokenizer, max_length=128, threshold=0.5,num_candidates=3):
    """
    For a given record, score all candidate sentences and return a dictionary
    of predicted gold indices (keys) and their sentences, if their probability is >= threshold.
    """
    # a=0
    # Get the question from the record.
    question = record["qa"]["question"]
    
    # Extract candidate sentences with their keys.
    candidates = extract_candidates(record)
    # print("candidates: ", candidates)
    # cand_json = json.dumps(candidates, indent=2)
    # print("candidates:",cand_json)
    # model.eval()  # Set the model in evaluation mode
    predicted_gold = {}

    # Loop over each candidate and compute its relevance score.
    with torch.no_grad():
        for key, sentence in candidates.items():
            # Prepare the input: concatenate question and candidate sentence.
            # input_text = question + " [SEP] " + sentence
            encoding = tokenizer.encode_plus(
                question,
                sentence,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            # encoding['input_ids'] = torch.clamp(encoding['input_ids'], 0,vocab_size - 1)  #vocab_size - 1 is the max valid index
            # Move input tensors to the appropriate device (assume same as model)

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Create segment_ids (assuming they are all 0)
            segment_ids = encoding["token_type_ids"].to(device)#torch.zeros_like(input_ids).to(device)
            # if(a==0):
            #   print(input_text)
            #   print(len(input_text.split(" ")))
            #   print("input_ids: ",input_ids)
            #   print("attention_mask: ",attention_mask)
            #   print("segment_ids: ",segment_ids)
            #   a=1
            # Get logits from the model, including segment_ids
            logits = model(input_ids, segment_ids, attention_mask)
            # print("logits: ",logits)
            probs = torch.softmax(logits, dim=1)
            # print("prob: ",probs)
            score = probs[0][1].item()  # probability for label "1" (relevant)
            # print("score: ",score)
            # If the candidate meets the threshold, add it to predicted gold inds.
            if score >= threshold:
                predicted_gold[key] = {
                    "sentence": sentence,
                    "score": score
                }
    predicted_gold=sorted(predicted_gold.items(), key=lambda item: item[1]['score'], reverse=True)[:num_candidates]
    print("predicted_gold: ",predicted_gold)
    return predicted_gold
