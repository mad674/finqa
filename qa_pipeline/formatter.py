import re

def deduplicate_sentences(sentences):
    seen = set()
    return [s for s in sentences if s not in seen and not seen.add(s)]

def split_into_sentences(text):
    # Split text into sentences using regex
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]



def merge_blocks(blocks, question):
    # Deduplicate and split sentences for pretext and posttext
    pretext = deduplicate_sentences(
        sum([split_into_sentences(" ".join(b["pretext"]) if isinstance(b["pretext"], list) else b["pretext"]) for b in blocks], [])
    )
    posttext = deduplicate_sentences(
        sum([split_into_sentences(" ".join(b["posttext"]) if isinstance(b["posttext"], list) else b["posttext"]) for b in blocks], [])
    )
    table = sum([b["table"] for b in blocks], [])  # Keep table as a list

    return {
        "question": question,
        "pretext": pretext,  # List of sentences
        "table": table,
        "posttext": posttext  # List of sentences
    }
    # Deduplicate and split sentences for pretext and posttext
    pretext = deduplicate_sentences(
        sum([split_into_sentences(b["pretext"]) for b in blocks], [])  # Apply sentence splitting directly
    )
    posttext = deduplicate_sentences(
        sum([split_into_sentences(b["posttext"]) for b in blocks], [])  # Apply sentence splitting directly
    )
    table = sum([b["table"] for b in blocks], [])  # Keep table as a list

    return {
        "question": question,
        "pretext": pretext,  # List of sentences
        "table": table,
        "posttext": posttext  # List of sentences
    }
    # Deduplicate and split sentences for pretext and posttext
    pretext = deduplicate_sentences(
        sum([split_into_sentences(" ".join(b["pretext"])) for b in blocks], [])
    )
    posttext = deduplicate_sentences(
        sum([split_into_sentences(" ".join(b["posttext"])) for b in blocks], [])
    )
    table = sum([b["table"] for b in blocks], [])  # Keep table as a list

    return {
        "question": question,
        "pretext": pretext,  # List of sentences
        "table": table,
        "posttext": posttext  # List of sentences
    }