from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
from transformers import BertTokenizerFast
import os
from pathlib import Path

cwd_parent = Path.absolute(Path.cwd()).parent

def train_tokenizer():
    print("=====Step1: train tokenizer=====")

    os.makedirs(f"{cwd_parent}/tokenizer_output", exist_ok=True)
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True
    )
    tokenizer.train(
        files=[f"{cwd_parent}/data/raw_data.txt"],
        vocab_size=30000,
        min_frequency=2
    )
    tokenizer.save_model(f"{cwd_parent}/tokenizer_output")
    print(f"Tokenizer trained and saved in {cwd_parent}/tokenizer_output/")




def process_and_load_dataset():
    print("=====Step2: load and process dataset=====")
    tokenizer = BertTokenizerFast(
        vocab_file=f"{cwd_parent}/tokenizer_output/vocab.txt",
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]"
    )
    def tokenize_function(examples):
        return tokenizer(
         examples['text'],
         padding='max_length',
         truncation=True,
         max_length=128,
         return_tensors="pt"
         )

    if not os.path.exists(f'{cwd_parent}/data/raw_data.txt'):
        raise FileNotFoundError(f"Training data file '{cwd_parent}/data/raw_data.txt' does not exist")
    if not os.path.exists(f'{cwd_parent}/data/val_data.txt'):
        raise FileNotFoundError(f"Validation data file '{cwd_parent}/data/val_data.txt' does not exist")
    train_dataset = load_dataset('text', data_files={'train': f'{cwd_parent}/data/raw_data.txt'}, split='train')
    val_dataset = load_dataset('text', data_files={'validation': f'{cwd_parent}/data/val_data.txt'}, split='validation')
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset.set_format("torch")
    return tokenizer, tokenized_train_dataset, tokenized_val_dataset

def save_datasets(train_dataset, val_dataset):
    os.makedirs(f"{cwd_parent}/data/tokenized_train", exist_ok=True)
    os.makedirs(f"{cwd_parent}/data/tokenized_val", exist_ok=True)
    train_dataset.save_to_disk(f"{cwd_parent}/data/tokenized_train")
    val_dataset.save_to_disk(f"{cwd_parent}/data/tokenized_val")
    print(f"Tokenized datasets saved in {cwd_parent}/data/tokenized_train and {cwd_parent}/data/tokenized_val")

if __name__ == '__main__':
    train_tokenizer()
    tokenizer, train_dataset,val_dataset = process_and_load_dataset()
    save_datasets(train_dataset,val_dataset)


