import sys, os
from tqdm import tqdm
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from data import MessageDataset, setup_tokenizer, get_json_files, split_dataset, collate_fn, create_input_target_pairs


# --- Data ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
setup_tokenizer(tokenizer)

DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", os.path.join(PROJECT_ROOT, "data"))
json_files = get_json_files(DATA_DIR)
full_dataset = MessageDataset(json_files, tokenizer=tokenizer, sequence_length=256)
train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
)

VOCAB_SIZE = len(tokenizer)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Vocab size: {VOCAB_SIZE}")
print(f"Dataset: {len(full_dataset)} total | {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")
print(f"Device: {DEVICE}")


# --- Model ---
class AttentionHead(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_sequence_length, max_sequence_length))
        )

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        d_k = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / d_k ** 0.5

        seq_len = x.size(-2)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return weights @ v


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_attention_heads, max_sequence_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.head_dim = d_model // num_attention_heads
        self.attention_heads = nn.ModuleList(
            AttentionHead(self.head_dim, max_sequence_length)
            for _ in range(num_attention_heads)
        )

        self.projection = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        out = self.norm1(x)

        head_output = []
        for i, head in enumerate(self.attention_heads):
            out_slice = out[:, :, i * self.head_dim: (i+1) * self.head_dim]
            head_output.append(head(out_slice))

        out = torch.cat(head_output, dim=-1)
        out = self.projection(out)

        x = x + out
        x = x + self.ff(self.norm2(x))

        return x


class GPT2Small(nn.Module):
    def __init__(self,
        vocab_size=VOCAB_SIZE,
        d_model=768,
        max_sequence_length=256,
        num_transformer_blocks=12,
        num_attention_heads=12
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_emb = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=d_model)

        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(d_model, num_attention_heads, max_sequence_length)
            for _ in range(num_transformer_blocks)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.size(1), device=x.device))

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = self.norm1(x)
        x = self.lm_head(x)

        return x


# --- Training ---
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", type=str, default=os.environ.get("SM_MODEL_DIR", "output"))
    args = parser.parse_args()

    # Rebuild dataloader with custom batch size
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    model = GPT2Small(
        d_model=args.d_model,
        max_sequence_length=args.seq_len,
        num_transformer_blocks=args.num_blocks,
        num_attention_heads=args.num_heads,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d={args.d_model}, blocks={args.num_blocks}, heads={args.num_heads} | {total_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            x, y = create_input_target_pairs(batch.to(DEVICE))
            pred = model(x)
            loss = loss_fn(pred.reshape(-1, pred.size(-1)), y.reshape(-1))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = create_input_target_pairs(batch.to(DEVICE))
                pred = model(x)
                val_loss += loss_fn(pred.reshape(-1, pred.size(-1)), y.reshape(-1)).item()

        train_avg = epoch_loss / len(dataloader)
        val_avg = val_loss / len(val_loader)
        cur_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["lr"].append(cur_lr)
        print(f"Epoch {epoch}: train={train_avg:.4f} val={val_avg:.4f} lr={cur_lr:.2e}")

    print(f"Model training complete in {time.time() - start:.1f}s")

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pt")
    torch.save(model.state_dict(), model_path)
    history_path = os.path.join(args.output, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")
