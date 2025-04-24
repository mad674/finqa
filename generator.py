import re, json, random, math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from transformers import get_linear_schedule_with_warmup
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use BERT for encoding
TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")  # ← FAST tokenizer

# ----------------------- Cell 2: Dataset, Vocabulary & Collate -----------------------
def filter_gold_inds(gold_inds):
    # Return only evidence segments containing digits.
    return {k: v for k, v in gold_inds.items() if any(c.isdigit() for c in v)}

def build_input(ex):
    question = ex['qa']['question']
    gold_entries = filter_gold_inds(ex['qa']['gold_inds'])
    # Sort evidence by digit count (descending), so numeric parts come first.
    gold_text = " ".join(sorted(gold_entries.values(), key=lambda s: -sum(ch.isdigit() for ch in s)))
    return question + "  " + gold_text

# Helper functions
def clean_token(tok):
    return tok.replace("Ġ", "").replace("▁", "").strip()

def is_number(s):
    return bool(re.match(r"^-?[\d,]*\.?\d+%?$", s))

def normalize_number_token(t):
    if t.endswith("%"):
        t = t[:-1]
    t = t.replace(",", "")
    try:
        if t.startswith("."):
            t = "0" + t  # add leading zero to .76 → 0.76
        if "." in t:
            t = str(float(t)).rstrip("0").rstrip(".")
        else:
            t = str(int(t))  # remove leading zeros
    except:
        return t
    return t
def align_number_token(raw_input, norm_target):
    """
    Attempt to find the starting character index of norm_target in raw_input.
    Tries both normalized and less strict variants to match the number in input.
    Returns -1 if not found.
    """
    try:
        # Try to find normalized number directly
        return raw_input.index(norm_target)
    except ValueError:
        # Try removing leading zeros (e.g., "0010" → "10")
        try:
            stripped = norm_target.lstrip("0")
            if stripped and stripped != norm_target:
                return raw_input.index(stripped)
        except ValueError:
            pass

        # Try matching a float-like pattern with optional leading 0
        try:
            if norm_target.startswith("0.") and len(norm_target) > 2:
                alt_form = norm_target[1:]  # try ".76" instead of "0.76"
                return raw_input.index(alt_form)
        except ValueError:
            pass

    # If none of the heuristics work
    print(f"❌ Failed to align number '{norm_target}' in input")
    return -1

def tokenize_dsl_line(line):
    # Tokenizes DSL line: subtract(959.2, 991.1) => ['subtract', '(', '959.2', ',', '991.1', ')']
    return re.findall(r"[a-zA-Z_]+|\#\d+|[(),]|[-+]?\d*\.\d+|\d+|<EOF>", line)


def parse_steps_with_memory(program: str):
    """
    Parses a program like: subtract(959.2, 991.1), divide(#0, 991.1)
    into stepwise tokens:
      [['subtract', '(', '959.2', ',', '991.1', ')'],
       ['divide', '(', '#0', ',', '991.1', ')'],
       ['<EOF>']]
    """
    steps = [['<SOS>']]
    program = program.strip()

    # Split on top-level commas between expressions
    depth = 0
    parts, start = [], 0
    for i, c in enumerate(program):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(program[start:i].strip())
            start = i + 1
    parts.append(program[start:].strip())

    # Tokenize each step
    for part in parts:
        if part:
            steps.append(tokenize_dsl_line(part))
    steps.append(['<EOF>'])
    return steps

def is_numeric_token(tok):
    return bool(re.match(r'^-?\d+(\.\d+)?$', tok.replace('##', ''))) or tok in ['.', ',']

def reconstruct_number_from_tokens(tokens, start_idx):
    while start_idx > 0 and tokens[start_idx].startswith('##'):
        start_idx -= 1

    number = ''
    i = start_idx

    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('##'):
            tok = tok[2:]

        if tok == '%' or tok == ',':
            number += tok
            i += 1
            continue

        if not (tok.isdigit() or tok == '.' or tok.replace('.', '', 1).isdigit()):
            break

        number += tok
        i += 1

    number = number.replace(",", "")
    number = re.sub(r'\.+$', '', number)

    # Allow numbers like "23.6%", "100%", etc.
    if not re.match(r'^-?\d+(\.\d+)?%?$', number):
        return None

    return number
def find_token_span_for_number(tokenizer, input_tokens, number_str):
    """
    Returns the index of the first token in input_tokens that matches
    the full subword sequence of number_str, or -1 if none.
    """
    number_tokens = tokenizer.tokenize(number_str)
    for i in range(len(input_tokens) - len(number_tokens) + 1):
        if input_tokens[i:i + len(number_tokens)] == number_tokens:
            return i
    return -1

class FinQADataset(Dataset):
    def __init__(self, data, tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"),op_file="operation_list.txt", const_file="constant_list.txt"):
        self.examples = []
        self.vocab = set()

        with open(op_file, 'r') as f:
            op_list = [line.strip() for line in f if line.strip()]
        with open(const_file, 'r') as f:
            const_list = [line.strip() for line in f if line.strip()]
        self.dsl_vocab = set(op_list + const_list + ['(', ')', ',', '<PAD>', '<COPY>', '<EOF>','<SOS>'])

        for ex in data:
            question = ex['qa']['question']
            context = " ".join([v for v in ex['qa']['gold_inds'].values() if any(c.isdigit() for c in v)])
            full_input = question + " " + context
            enc = tokenizer(full_input, return_tensors='pt', padding='max_length', truncation=True,
                            max_length=512, return_offsets_mapping=True)
            offsets = enc['offset_mapping'].squeeze(0).tolist()
            input_ids = enc['input_ids'].squeeze(0).to(device)
            input_mask = enc['attention_mask'].squeeze(0).to(device)
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            raw_input = full_input.lower()
            # tgt_lines = parse_steps_from_program(ex['qa']['program'])
            # tgt_tokens = [tok for line in tgt_lines for tok in tokenize_dsl_line(line)]
            tgt_lines = parse_steps_with_memory(ex['qa']['program'])  # custom memory-aware parser
            tgt_tokens = [tok for step in tgt_lines for tok in step]

            copy_labels = []
            for t in tgt_tokens:
                if t in self.dsl_vocab:
                    copy_labels.append(-100)
                elif is_number(t):
                    norm = normalize_number_token(t)
                    const_candidate = f"CONST_{norm}"
                    if const_candidate in self.dsl_vocab:
                        copy_labels.append(-100)
                        continue

                    token_idx = -1
                    start_char = align_number_token(raw_input, norm)
                    if start_char != -1:
                        for i, (start, end) in enumerate(offsets):
                            if start <= start_char < end:
                                token_idx = i
                                break

                    # Fallback to token-based number reconstruction
                    if token_idx == -1:
                        for i in range(len(input_tokens)):
                            reconstructed = reconstruct_number_from_tokens(input_tokens, i)
                            if reconstructed and abs(float(reconstructed) - float(norm)) < 1e-4:
                                token_idx = i
                                break

                    if token_idx != -1:
                        copy_labels.append(token_idx)
                    else:
                        print(f"❌ Could not map '{norm}' to token index")
                        copy_labels.append(-100)
                else:
                    copy_labels.append(-100)

            self.vocab.update([tok for tok in self.dsl_vocab])
            self.examples.append({
                'input_ids': input_ids,
                'input_mask': input_mask,
                'input_tokens': input_tokens,
                'program_tokens': tgt_tokens,
                'copy_labels': copy_labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Collate + Vocab

def build_vocab(vocab_set):
    vocab = sorted(list(vocab_set | {"<PAD>", "<EOF>", "<COPY>","<SOS>"}))
    return vocab, {tok: i for i, tok in enumerate(vocab)}

def collate_fn(batch):
    max_len = max(len(x['program_tokens']) for x in batch)
    for x in batch:
        x['program_tokens'] += ["<PAD>"] * (max_len - len(x['program_tokens']))
        x['copy_labels'] += [-100] * (max_len - len(x['copy_labels']))
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]).to(device),
        'input_mask': torch.stack([x['input_mask'] for x in batch]).to(device),
        'input_tokens': [x['input_tokens'] for x in batch],
        'program_tokens': [x['program_tokens'] for x in batch],
        'copy_labels': torch.tensor([x['copy_labels'] for x in batch]).to(device)
    }


class PointerDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, vocab_dict):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.vocab_proj = nn.Linear(hidden_size, vocab_size)
        self.copy_proj = nn.Linear(hidden_size, hidden_size)
        self.enc_proj = nn.Linear(768, hidden_size)
        self.vocab_dict = vocab_dict

    def forward(self, dec_input, enc_output, hidden, input_tokens, input_mask):
        # ✅ Add sanity check to avoid index errors
        if dec_input.max().item() >= self.emb.num_embeddings:
            raise ValueError(f"Invalid index in dec_input: {dec_input.max().item()} >= vocab_size {self.emb.num_embeddings}")

        embedded = self.emb(dec_input)
        output, hidden = self.lstm(embedded, hidden)
        vocab_logits = self.vocab_proj(output)

        enc_proj = self.enc_proj(enc_output)
        attn_scores = torch.bmm(self.copy_proj(output), enc_proj.transpose(1, 2))

        special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}
        mask = input_mask.clone()
        for b in range(len(input_tokens)):
            for i, tok in enumerate(input_tokens[b]):
                if tok in special_tokens:
                    mask[b, i] = 0

        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        if input_tokens is not None:
            for b in range(len(input_tokens)):
                for i, tok in enumerate(input_tokens[b]):
                    if is_number(clean_token(tok)):
                        attn_scores[b, 0, i] += 2.0

        attn_weights = torch.softmax(attn_scores, dim=-1)
        return vocab_logits, attn_weights, hidden

class PointerProgramGenerator(nn.Module):
    def __init__(self, vocab_dict):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        # Use hidden_size = 768 (BERT hidden size), emb_size = 256
        self.decoder = PointerDecoder(hidden_size=768, vocab_size=len(vocab_dict), emb_size=256, vocab_dict=vocab_dict)
        self.vocab_dict = vocab_dict

    def forward(self, input_ids, input_mask, tgt_ids=None, input_tokens=None, max_len=20, return_attention=False):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=input_mask).last_hidden_state  # [B, T, 768]
        batch_size = input_ids.size(0)
        dec_input = torch.full((batch_size, 1), self.vocab_dict["<SOS>"], dtype=torch.long, device=device)
        hidden, logits_all, attns_all = None, [], []
        decode_len = tgt_ids.size(1) if tgt_ids is not None else max_len

        if tgt_ids is None:
            # track which sequences have already generated <EOF>
            finished = torch.zeros(batch_size, dtype=torch.bool, device=dec_input.device)

        for t in range(decode_len):
            vocab_logits, attn_weights, hidden = self.decoder(dec_input, enc_out, hidden, input_tokens, input_mask)
            logits_all.append(vocab_logits)  # [B, 1, V]
            attns_all.append(attn_weights)     # [B, 1, T]
            if tgt_ids is not None:
                dec_input = tgt_ids[:, t].unsqueeze(1)
            else:
                dec_input = vocab_logits.argmax(-1).unsqueeze(1)
                is_eos = dec_input.squeeze(1) == self.vocab_dict["<EOF>"]
                finished |= is_eos
                if finished.all():
                    break
        logits_cat = torch.cat(logits_all, dim=1)  # [B, L, V]
        if return_attention:
            attns_cat = torch.cat(attns_all, dim=1)  # [B, L, T]
            return logits_cat, attns_cat
        return logits_cat
    
    
def encode_programs(program_tokens, vocab_dict):
    max_len = max(len(p) for p in program_tokens)
    tensor = torch.full((len(program_tokens), max_len), vocab_dict["<PAD>"], dtype=torch.long, device=device)
    for i, tokens in enumerate(program_tokens):
        for j, tok in enumerate(tokens):
            if tok in vocab_dict:
                tensor[i, j] = vocab_dict[tok]
            elif is_number(tok):
                tensor[i, j] = vocab_dict["<COPY>"]
            else:
                tensor[i, j] = vocab_dict.get(tok, vocab_dict["<COPY>"])  # fallback

    return tensor



def train(model, dataloader, vocab_dict, epochs=5, lr=1e-5,ptr_weight = 0.5, patience=3):
    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=len(dataloader) * epochs)
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab_dict["<PAD>"])
    ptr_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            input_tokens = batch["input_tokens"]
            copy_labels = batch["copy_labels"].to(device)
            tgt = encode_programs(batch["program_tokens"], vocab_dict).to(input_ids.device)

            logits, attns = model(input_ids, input_mask, tgt, input_tokens, return_attention=True)
            logits = logits[:, :tgt.size(1), :]
            attns = attns[:, :tgt.size(1), :]

            loss1 = ce_loss(logits.view(-1, logits.size(-1)), tgt.view(-1))

            copy_mask = (tgt == vocab_dict["<COPY>"])
            if copy_mask.any():
                attn_flat = attns.view(-1, attns.size(-1))
                label_flat = batch["copy_labels"].view(-1)

                valid_ptr_mask = label_flat != -100
                if valid_ptr_mask.any():
                    loss2_all = ptr_loss_fn(attn_flat[valid_ptr_mask], label_flat[valid_ptr_mask])
                    loss2 = loss2_all.mean()
                else:
                    loss2 = torch.tensor(0.0, device=device)
            else:
                loss2 = torch.tensor(0.0, device=device)

            loss = loss1 + ptr_weight * loss2

            if not torch.isfinite(loss):
                print("⚠ NaN loss, skipping")
                continue

            non_pad_mask = (tgt != vocab_dict["<PAD>"])
            pred_tokens = logits.argmax(-1)
            correct_tokens = (pred_tokens == tgt) & non_pad_mask
            total_correct += correct_tokens.sum().item()
            total_tokens += non_pad_mask.sum().item()

            if ep % 1 == 0 and batch_idx == 0:
                inv_vocab = {v: k for k, v in vocab_dict.items()}
                for i in range(min(3, len(pred_tokens))):
                    pred_program = [inv_vocab[t.item()] for t in pred_tokens[i]]
                    true_program = batch["program_tokens"][i]
                    print(f"Epoch {ep+1}, Batch {batch_idx+1}, Example {i+1}")
                    print(f"Predicted: {pred_program}")
                    print(f"True: {true_program}")
                    print("-" * 40)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens * 100
        print(f"Epoch {ep+1}/{epochs} Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}% | Batch Time: {time.time() - start_time:.2f}s")
        start_time = time.time()

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠ No improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹ Early stopping triggered.")
                break


# def infer(model, sample, vocab_dict, max_len=50):
#     model.eval()
#     inv_vocab = {v: k for k, v in vocab_dict.items()}

#     input_ids = sample["input_ids"].unsqueeze(0).to(device)
#     input_mask = sample["input_mask"].unsqueeze(0).to(device)
#     input_tokens = [sample["input_tokens"]]

#     outputs, hidden = [], None
#     memory, step_idx = {}, 0
#     curr_tokens = []

#     try:
#         sos_id = int(vocab_dict["<SOS>"])
#         eof_id = int(vocab_dict["<EOF>"])
#         dec_input = torch.full((1, 1), sos_id, dtype=torch.long).to(device)
#         print("✅ '<SOS>' token ID:", sos_id)
#         print("✅ '<EOF>' token ID:", eof_id)
#     except Exception as e:
#         print("❌ Decoder init error:", e)
#         return []


#     step_type = 0  # 0: op, 1: (, 2: arg1, 3: ,, 4: arg2, 5: ), 6: <EOF>
#     with torch.no_grad():
#         enc_out = model.encoder(input_ids=input_ids, attention_mask=input_mask).last_hidden_state

#         for step in range(max_len):
#             vocab_logits, attn_weights, hidden = model.decoder(
#                 dec_input, enc_out, hidden, input_tokens, input_mask
#             )
#             logits = vocab_logits[0, 0]  # shape: [vocab_size]

#             # Grammar forcing
#             allowed = []
#             for tok, idx in vocab_dict.items():
#                 if step_type == 0 and tok not in {"add", "subtract", "multiply", "divide","exp","greater","table_sum","table_average","table_max","table_min"}:
#                     continue
#                 elif step_type == 1 and tok != "(":
#                     continue
#                 elif step_type == 2 and not (tok.startswith("CONST_") or tok.startswith("#") or is_number(tok) or tok == "<COPY>"):
#                     continue
#                 elif step_type == 3 and tok != ",":
#                     continue
#                 elif step_type == 4 and not (tok.startswith("CONST_") or tok.startswith("#") or is_number(tok) or tok == "<COPY>"):
#                     continue
#                 elif step_type == 5 and tok != ")":
#                     continue
#                 elif step_type == 6:
#                     if tok != "<EOF>":
#                         continue
#                     if step_idx < 1:
#                         continue  # ⛔ prevent early EOF
#                 allowed.append(idx)

#             if allowed:
#                 mask = torch.full_like(logits, float("-inf"))
#                 for idx in allowed:
#                     if idx < logits.size(0):
#                         mask[idx] = 0
#                 logits = logits + mask

#             pred_token_id = logits.argmax(-1).item()
#             pred_token = inv_vocab.get(pred_token_id, "<UNK>")
#             print(f"\n📘 Step {step+1} — Predicted: {pred_token}")

#             if pred_token == "<EOF>":
#                 print("🛑 <EOF> reached")
#                 break

#             elif pred_token == "<COPY>":
#                 attn = attn_weights.squeeze(0).squeeze(0)
#                 pointer_idx = attn.argmax().item()

#                 print("📈 Top-5 Attention Scores for Numeric Tokens:")
#                 numeric_scores = [(clean_token(input_tokens[0][i]), attn[i].item())
#                                   for i in range(len(attn))
#                                   if is_number(clean_token(input_tokens[0][i]))]
#                 for tok, score in sorted(numeric_scores, key=lambda x: -x[1])[:5]:
#                     print(f"{tok:>10} | {score:.4f}")

#                 if 0 <= pointer_idx < len(input_tokens[0]):
#                     copied_raw = input_tokens[0][pointer_idx]
#                     try:
#                         reconstructed = reconstruct_number_from_tokens(input_tokens[0], pointer_idx)
#                     except:
#                         reconstructed = None
#                     curr_tokens.append(reconstructed if reconstructed else clean_token(copied_raw))
#                 else:
#                     curr_tokens.append("<UNK>")

#             else:
#                 if(pred_token!="<SOS>"):
#                 # Auto-fill memory references
#                     if step_type in {2, 4} and pred_token.startswith("#"):
#                         curr_tokens.append(f"#{step_idx - 1}" if step_idx > 0 else pred_token)
#                     else:
                        
#                         curr_tokens.append(pred_token)

#             if pred_token == ")":
#                 step_line = " ".join(curr_tokens)
#                 outputs.append(step_line)
#                 memory[step_idx] = step_line
#                 step_idx += 1
#                 curr_tokens = []
#                 if step_idx >= 1:
#                     curr_tokens.append(",")
#             # Update decoder input
#             dec_input = torch.tensor([[pred_token_id]], dtype=torch.long).to(device)
#             step_type = (step_type + 1) % 7
#             if step_type == 6:
#               if tok != "<EOF>":
#                   continue
#               if step_idx < 1:  # ⛔ safeguard
#                   continue


#     return outputs or [" ".join(curr_tokens)]



def infer(model, sample, vocab_dict, max_len=50):
    model.eval()
    inv_vocab = {v: k for k, v in vocab_dict.items()}

    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    input_mask = sample["input_mask"].unsqueeze(0).to(device)
    input_tokens = [sample["input_tokens"]]

    outputs, hidden = [], None
    memory, step_idx = {}, 0
    curr_tokens = []

    try:
        sos_id = int(vocab_dict["<SOS>"])
        eof_id = int(vocab_dict["<EOF>"])
        dec_input = torch.full((1, 1), sos_id, dtype=torch.long).to(device)
        print("✅ '<SOS>' token ID:", sos_id)
        print("✅ '<EOF>' token ID:", eof_id)
    except Exception as e:
        print("❌ Decoder init error:", e)
        return []


    step_type = 0  # 0: op, 1: (, 2: arg1, 3: ,, 4: arg2, 5: ), 6: <EOF>
    with torch.no_grad():
        enc_out = model.encoder(input_ids=input_ids, attention_mask=input_mask).last_hidden_state

        for step in range(max_len):
            vocab_logits, attn_weights, hidden = model.decoder(
                dec_input, enc_out, hidden, input_tokens, input_mask
            )
            logits = vocab_logits[0, 0]  # shape: [vocab_size]

            pred_token_id = logits.argmax(-1).item()
            pred_token = inv_vocab.get(pred_token_id, "<UNK>")
            print(f"\n📘 Step {step+1} — Predicted: {pred_token}")

            if pred_token == "<EOF>":
                print("🛑 <EOF> reached")
                break

            elif pred_token == "<COPY>":
                attn = attn_weights.squeeze(0).squeeze(0)
                pointer_idx = attn.argmax().item()

                print("📈 Top-5 Attention Scores for Numeric Tokens:")
                numeric_scores = [(clean_token(input_tokens[0][i]), attn[i].item())
                                  for i in range(len(attn))
                                  if is_number(clean_token(input_tokens[0][i]))]
                for tok, score in sorted(numeric_scores, key=lambda x: -x[1])[:5]:
                    print(f"{tok:>10} | {score:.4f}")

                if 0 <= pointer_idx < len(input_tokens[0]):
                    copied_raw = input_tokens[0][pointer_idx]
                    try:
                        reconstructed = reconstruct_number_from_tokens(input_tokens[0], pointer_idx)
                    except:
                        reconstructed = None
                    curr_tokens.append(reconstructed if reconstructed else clean_token(copied_raw))
                else:
                    curr_tokens.append("<UNK>")

            else:
                if(pred_token!="<SOS>"):
                    # Auto-fill memory references
                    if step_type in {2, 4} and pred_token.startswith("#"):
                        curr_tokens.append(f"#{step_idx - 1}" if step_idx > 0 else pred_token)
                    else:
                        curr_tokens.append(pred_token)

            if pred_token == ")":
                step_line = " ".join(curr_tokens)
                outputs.append(step_line)
                memory[step_idx] = step_line
                step_idx += 1
                curr_tokens = []
                if step_idx >= 1:
                    curr_tokens.append(",")

            dec_input = torch.tensor([[pred_token_id]], dtype=torch.long).to(device)
            step_type = (step_type + 1) % 7
            if step_type == 6:
              if tok != "<EOF>":
                  continue
              if step_idx < 1:  # ⛔ safeguard
                  continue


    return outputs or [" ".join(curr_tokens)]
