import pdfplumber
import re

def split_into_sentences(text):
    cleaned = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in sentences if s.strip()]

def table_to_sentences(table):
    if not table or len(table) < 2:
        return []
    headers = table[0]
    return [
        ", ".join(f"{headers[i]}: {cell}" for i, cell in enumerate(row if row else [])) + "."
        for row in table[1:]
    ]

def extract_blocks_from_pdf(pdf_path):
    blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            tables = page.extract_tables()

            if not page_text or not tables:
                continue

            for idx, table in enumerate(tables):
                pretext = split_into_sentences(page_text[:200])
                posttext = split_into_sentences(page_text[-200:])
                table_clean = [
                    [cell if cell is not None else "" for cell in row]
                    for row in table if any(row)
                ]
                block = {
                    "id": f"{pdf_path}-p{page_num}-t{idx}",
                    "source": {"pdf": pdf_path, "page": page_num, "table_index": idx},
                    "pretext": pretext,
                    "posttext": posttext,
                    "table": table_clean,
                    "table_sentences": table_to_sentences(table_clean)
                }
                blocks.append(block)

    return blocks
