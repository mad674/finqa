import re

def deduplicate_sentences(sentences):
    seen = set()
    return [s for s in sentences if s not in seen and not seen.add(s)]

def split_into_sentences(text):
    # Split text into sentences using regex
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]



def merge_blocks(blocks, question):
    """
    Merges sentences from blocks into pretext, posttext, and table.
    Each block is expected to have "pretext", "posttext", and "table" keys.
    """
    try:
        # Deduplicate and split sentences for pretext
        pretext = deduplicate_sentences(
            sum([
                split_into_sentences(
                    " ".join(b["pretext"]) if isinstance(b, dict) and isinstance(b.get("pretext"), list)
                    else (b["pretext"] if isinstance(b, dict) and isinstance(b.get("pretext"), str) else "")
                )
                for b in blocks
            ], [])
        )

        # Deduplicate and split sentences for posttext
        posttext = deduplicate_sentences(
            sum([
                split_into_sentences(
                    " ".join(b["posttext"]) if isinstance(b, dict) and isinstance(b.get("posttext"), list)
                    else (b["posttext"] if isinstance(b, dict) and isinstance(b.get("posttext"), str) else "")
                )
                for b in blocks
            ], [])
        )

        # Combine tables
        table = sum([b["table"] for b in blocks if isinstance(b, dict) and "table" in b], [])

        return {
            "question": question,
            "pretext": pretext,  # List of sentences
            "table": table,      # List of table rows
            "posttext": posttext  # List of sentences
        }

    except Exception as e:
        print(f"Error in merge_blocks: {e}")
        return {
            "question": question,
            "pretext": [],
            "table": [],
            "posttext": []
        }