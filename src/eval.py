import torch
import math
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BertTokenizerFast
import os
import torch.nn as nn
from torch.nn import functional as F
from model import MyLLM
from data_process import process_and_load_dataset,cwd_parent


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_model(model, val_dataset_loader):
    total_val_loss = 0
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in val_dataset_loader:
            input_ids = batch['input_ids'].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            logits = model(inputs)

            # 使用 reshape 而不是 view
            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataset_loader)
    perplexity = math.exp(avg_val_loss)
    return avg_val_loss, perplexity

if __name__ == '__main__':
    tokenizer, _, val_dataset = process_and_load_dataset()
    val_dataset_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 512,
        "block_size": 128,
        "num_heads": 8,
        "num_layers": 6
    }
    model = MyLLM(**MODEL_CFG)
    state_dict = torch.load(f"{cwd_parent}/checkpoints/pretrained_llm.pth", map_location=device)
    keys_to_remove = [k for k in state_dict.keys() if 'tril' in k]
    for k in keys_to_remove:
        del state_dict[k]

    # 加载状态字典，允许不严格匹配
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    avg_val_loss, perplexity = evaluate_model(model, val_dataset_loader)
    print(f"验证集平均损失: {avg_val_loss:.4f}")
    print(f"困惑度 (Perplexity): {perplexity:.2f}")