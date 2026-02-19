import argparse
import glob
import json
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


# ── Dataset ──────────────────────────────────────────────────────────────────

class MessageDataset(Dataset):
    def __init__(self, json_files, tokenizer, sequence_length=512):
        self.sequences = []
        self.speaker_to_id = {}
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        # First pass: collect all unique speakers
        all_speakers = set()
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for msg in data.get("messages", []):
                    if (
                        "content" in msg
                        and not self._is_reaction(msg["content"])
                        and not self._contains_emoji(msg["content"])
                    ):
                        all_speakers.add(msg["sender_name"])

        # Assign speaker IDs
        for idx, speaker in enumerate(sorted(all_speakers), start=1):
            self.speaker_to_id[speaker] = f"<S{idx}>"

        # Second pass: build conversation sequences
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                valid_messages = [
                    msg
                    for msg in data.get("messages", [])
                    if (
                        "content" in msg
                        and not self._is_reaction(msg["content"])
                        and not self._contains_emoji(msg["content"])
                    )
                ]
                self._create_sequences(valid_messages)

    def _create_sequences(self, messages):
        current_sequence = []
        current_length = 0

        for msg in messages:
            speaker_token = self.speaker_to_id[msg["sender_name"]]
            cleaned_text = self._clean_text(msg["content"])
            formatted_msg = f"{speaker_token} {cleaned_text} <EOM>"

            msg_tokens = self.tokenizer.encode(formatted_msg)
            msg_length = len(msg_tokens)

            if current_length + msg_length > self.sequence_length and current_sequence:
                self.sequences.append("".join(current_sequence))
                current_sequence = []
                current_length = 0

            current_sequence.append(formatted_msg)
            current_length += msg_length

        if current_sequence:
            self.sequences.append("".join(current_sequence))

    @staticmethod
    def _is_reaction(text):
        return "reacted" in text.lower() and "to your message" in text.lower()

    @staticmethod
    def _contains_emoji(text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U00002600-\U000026FF"
            "\U00002B50"
            "]+",
            flags=re.UNICODE,
        )
        return bool(emoji_pattern.search(text))

    @staticmethod
    def _clean_text(text):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        text = " ".join(text.split())
        text = re.sub(r"http\S+|www\S+", "", text)
        return text.strip()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(
            self.sequences[idx],
            max_length=self.sequence_length,
            truncation=True,
        )
        return torch.tensor(encoded, dtype=torch.long)


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))

        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=256, n_heads=8,
                 n_layers=6, d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)

        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )
        return logits, loss


# ── Helpers ──────────────────────────────────────────────────────────────────

def collate_fn(batch, pad_token_id):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq
    return padded


def create_input_target_pairs(batch):
    return batch[:, :-1], batch[:, 1:]


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLTSM transformer on message data")

    # SageMaker environment variables with local-friendly defaults
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "data"),
                        help="Directory containing JSON message files")
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "model_output"),
                        help="Directory to save model artifacts")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"),
                        help="Directory for additional output artifacts")

    # Hyperparameters (passed by SageMaker as CLI args)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "additional_special_tokens": [
            "<S1>", "<S2>", "<S3>", "<S4>", "<S5>",
            "<S6>", "<S7>", "<S8>", "<EOM>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # ── Discover data files ──────────────────────────────────────────────
    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.data_dir}")
    print(f"Found {len(json_files)} JSON files in {args.data_dir}")

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    dataset = MessageDataset(json_files, tokenizer=tokenizer, sequence_length=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0,
    )
    print(f"Dataset size: {len(dataset)} sequences | {len(dataloader)} batches")

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ── Model ────────────────────────────────────────────────────────────
    model = Transformer(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Training ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            inputs, targets = create_input_target_pairs(batch)
            inputs, targets = inputs.to(device), targets.to(device)

            logits, loss = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── Save model artifacts ─────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "speaker_to_id": dataset.speaker_to_id,
        "args": vars(args),
    }, os.path.join(args.model_dir, "model.pt"))

    # Save tokenizer so it can be reloaded for inference
    tokenizer.save_pretrained(args.model_dir)

    print(f"Model artifacts saved to {args.model_dir}")


if __name__ == "__main__":
    main()
