# JeevPT: Building a Small Language Model

## Overview
- What is JeevPT — a custom transformer trained on group chat messages to mimic conversational style
- Why build from scratch instead of fine-tuning an existing model
- End result: a 17M parameter model that generates group chat messages in the style of specific speakers

## Scope
- Goal: next-token prediction on multi-speaker conversation data
- Not a general-purpose LLM — scoped to a specific friend group's messaging patterns
- Custom speaker tokens (`<S1>`, `<S2>`, ...) to distinguish who's talking
- Built with PyTorch, trained on AWS SageMaker

## Data Sources
- Facebook Messenger export (JSON format)
- ~6 conversation files across multiple group chats
- ~415K tokens, 829 sequences after processing

## Data Preparation
- Filtering: removed reactions ("reacted to your message"), emoji-only messages, URLs
- Text cleaning: fixed mojibake encoding (latin-1 → utf-8), normalized whitespace
- Speaker mapping: assigned each unique speaker a special token ID
- Sequence construction: concatenated messages into 512-token chunks with `<EOM>` delimiters
- Tokenization: GPT-2 BPE tokenizer with 9 added special tokens (8 speakers + `<EOM>`)

## Modeling
- Architecture: decoder-only transformer (GPT-style)
  - Token + learned positional embeddings
  - 6 transformer blocks with pre-norm (LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN → Residual)
  - 8 attention heads, 256 embedding dim, 1024 FFN dim
  - Causal mask for autoregressive generation
  - Weight tying between token embeddings and output head
- Total parameters: 17.7M
- Vocab size: 50,266 (GPT-2 base + special tokens)

## Training
- Objective: next-token prediction with cross-entropy loss (padding tokens ignored)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: cosine annealing over 50 epochs
- Gradient clipping: max norm 1.0
- Batch size: 8, sequence length: 512
- Infrastructure: AWS SageMaker, ml.g4dn.xlarge (NVIDIA T4, 16GB VRAM)
- Training time and cost

## Eval
- Training loss curve over 50 epochs
- Qualitative evaluation: generated conversation samples
- Does the model capture speaker-specific patterns and slang?
- Limitations: small dataset, no validation split, risk of memorization

## Conclusion
- What worked, what didn't
- What you'd do differently (dataset size, tokenizer choice, architecture tweaks)
- Next steps
