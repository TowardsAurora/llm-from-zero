import torch
from torch.utils.data import DataLoader,Dataset
from model import MyLLM
from data_process import process_and_load_dataset,cwd_parent
from tqdm import tqdm
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InstructionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"指令: {item['instruction']}\n输入: {item.get('input', '')}\n响应: {item['response']}"
        tokenized_output = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized_output['input_ids'].squeeze(0),
            'attention_mask': tokenized_output['attention_mask'].squeeze(0),
        }

def instruction_turning(num_epoches=4, model=None, dataloader=None, optimizer=None, loss_function=None):
    model.train()
    for epoch in range(num_epoches):
        loop = tqdm(dataloader,total=len(dataloader),desc=f'Epoch {epoch + 1}/{num_epoches}')
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            outputs = model(inputs)
            loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(),avg_loss=total_loss/(loop.n+1))
    print("finish pre-training...")
    return model

if __name__ == '__main__':
    tokenizer, _, _ = process_and_load_dataset()
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

    instruction_data_path = f"{cwd_parent}/data/instruction_data.jsonl"
    instruction_dataset = InstructionDataset(instruction_data_path, tokenizer, max_length=MODEL_CFG.get("block_size"))
    dataloader = DataLoader(instruction_dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    turning_model = instruction_turning(4, model, dataloader, optimizer, loss_function)
    print(f"model saved at {cwd_parent}/checkpoints/instruction_turning_llm.pth")
    torch.save(turning_model.state_dict(), f"{cwd_parent}/checkpoints/instruction_turning_llm.pth")

