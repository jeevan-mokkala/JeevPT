import sys, os
from tqdm import tqdm
import time
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from data import MessageDataset, setup_tokenizer, get_json_files, split_dataset, collate_fn, create_input_target_pairs


# --- Data ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
setup_tokenizer(tokenizer)

DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", os.path.join(PROJECT_ROOT, "data"))
json_files = get_json_files(DATA_DIR)
full_dataset = MessageDataset(json_files, tokenizer=tokenizer, sequence_length=256)
train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Vocab size: {len(tokenizer)}")
print(f"Dataset: {len(full_dataset)} total | {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")
print(f"Device: {DEVICE}")


# --- LoRA ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=2):
        super().__init__()
        self.original_layer = original_layer

        in_shape = original_layer.weight.shape[0]
        out_shape = original_layer.weight.shape[1]

        self.A = nn.Parameter(torch.empty(in_shape, r))
        nn.init.kaiming_uniform_(self.A)
        self.B = nn.Parameter(torch.zeros(r, out_shape))

    def forward(self, x):
        return self.original_layer(x) + x @ self.A @ self.B


# --- Training ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default=os.environ.get("SM_MODEL_DIR", "output"))
    args = parser.parse_args()

    # Load pretrained GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace layers with LoRA wrappers
    for block in model.transformer.h:
        block.attn.c_attn = LoRALayer(block.attn.c_attn, r=args.rank)
        block.mlp.c_fc = LoRALayer(block.mlp.c_fc, r=args.rank)
        block.mlp.c_proj = LoRALayer(block.mlp.c_proj, r=args.rank)

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.4f}M ({trainable_params/total_params*100:.2f}%) | rank={args.rank}")

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            x, y = create_input_target_pairs(batch.to(DEVICE))
            pred = model(x).logits
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
                pred = model(x).logits
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

    # Save only LoRA weights
    lora_state = {k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
    lora_path = os.path.join(args.output, "lora_weights.pt")
    torch.save(lora_state, lora_path)
    print(f"LoRA weights saved to {lora_path} ({len(lora_state)} tensors)")

    history_path = os.path.join(args.output, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"History saved to {history_path}")
