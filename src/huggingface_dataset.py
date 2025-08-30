from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
from transformers import BertTokenizerFast
import os
from pathlib import Path

cwd_parent = Path.absolute(Path.cwd()).parent


def load_huggingface_dataset(dataset_name="wikitext", dataset_config="wikitext-103-raw-v1"):
    """从HuggingFace加载数据集并保存为文本文件"""
    print(f"=====Loading {dataset_name} dataset from HuggingFace=====")

    # 加载数据集
    dataset = load_dataset(dataset_name, dataset_config)

    # 保存训练集
    train_texts = []
    for example in dataset["train"]:
        if example["text"].strip():  # 跳过空文本
            train_texts.append(example["text"])

    with open(f"{cwd_parent}/data/raw_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))

    # 保存验证集
    val_texts = []
    for example in dataset["validation"]:
        if example["text"].strip():  # 跳过空文本
            val_texts.append(example["text"])

    with open(f"{cwd_parent}/data/val_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_texts))

    print(f"Dataset saved to {cwd_parent}/data/")


def train_tokenizer():
    """训练tokenizer"""
    print("=====Step1: train tokenizer=====")

    # 确保数据存在
    if not os.path.exists(f"{cwd_parent}/data/raw_data.txt"):
        load_huggingface_dataset()

    os.makedirs(f"{cwd_parent}/tokenizer_output", exist_ok=True)
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    tokenizer.train(
        files=[f"{cwd_parent}/data/raw_data.txt"],
        vocab_size=50000,  # 增加词汇表大小
        min_frequency=2,
        limit_alphabet=1000,
        wordpieces_prefix="##"
    )
    tokenizer.save_model(f"{cwd_parent}/tokenizer_output")
    print(f"Tokenizer trained and saved in {cwd_parent}/tokenizer_output/")


def process_and_load_dataset():
    """处理并加载数据集"""
    print("=====Step2: load and process dataset=====")

    # 确保tokenizer已训练
    if not os.path.exists(f"{cwd_parent}/tokenizer_output/vocab.txt"):
        train_tokenizer()

    tokenizer = BertTokenizerFast(
        vocab_file=f"{cwd_parent}/tokenizer_output/vocab.txt",
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,  # 增加序列长度
            return_tensors="pt"
        )

    # 加载数据集
    train_dataset = load_dataset('text', data_files={'train': f'{cwd_parent}/data/raw_data.txt'}, split='train')
    val_dataset = load_dataset('text', data_files={'validation': f'{cwd_parent}/data/val_data.txt'}, split='validation')

    # 分词处理
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4)

    # 设置格式
    tokenized_train_dataset.set_format("torch", columns=['input_ids', 'attention_mask'])
    tokenized_val_dataset.set_format("torch", columns=['input_ids', 'attention_mask'])

    return tokenizer, tokenized_train_dataset, tokenized_val_dataset


if __name__ == '__main__':
    # 下载数据集
    load_huggingface_dataset()

    # 训练tokenizer
    train_tokenizer()

    # 处理数据集
    tokenizer, train_dataset, val_dataset = process_and_load_dataset()
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")