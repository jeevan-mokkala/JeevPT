import glob
import json
import os
import re
import torch
from torch.utils.data import Dataset


class MessageDataset(Dataset):
    def __init__(self, json_files, tokenizer, sequence_length=256):
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

            if msg_length > self.sequence_length:
                continue

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
        encoded = self.tokenizer.encode(self.sequences[idx])
        return torch.tensor(encoded, dtype=torch.long)


SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<S1>", "<S2>", "<S3>", "<S4>", "<S5>",
        "<S6>", "<S7>", "<S8>", "<EOM>",
    ]
}


def setup_tokenizer(tokenizer):
    """Add special tokens and set pad token. Returns number of tokens added."""
    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.pad_token = tokenizer.eos_token
    return num_added


def get_json_files(data_dir="data"):
    json_files = sorted(glob.glob(os.path.join(data_dir, "**/message*.json"), recursive=True))
    if not json_files:
        raise FileNotFoundError(f"No message JSON files found in {data_dir}")
    return json_files


def split_dataset(dataset, train=0.8, val=0.1, test=0.1, seed=42):
    from torch.utils.data import random_split
    n = len(dataset)
    train_n = int(n * train)
    val_n = int(n * val)
    test_n = n - train_n - val_n
    return random_split(dataset, [train_n, val_n, test_n],
                        generator=torch.Generator().manual_seed(seed))


def collate_fn(batch, pad_token_id):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq
    return padded


def create_input_target_pairs(batch):
    return batch[:, :-1], batch[:, 1:]
