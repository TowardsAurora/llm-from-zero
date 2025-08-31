import torch
from torch.utils.data import DataLoader
from new_model import MyLLM
from huggingface_dataset import process_and_load_dataset, cwd_parent
from tqdm import tqdm
import math
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


def pretrain(num_epochs, model, train_data_loader, val_data_loader, optimizer, loss_function, scheduler=None):
    """预训练函数"""
    print("Starting pre-training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_loop = tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            # 前向传播
            outputs = model(inputs)
            loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.reshape(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (train_loop.n + 1)
            train_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        # 验证阶段
        val_loss, perplexity = evaluate_model(model, val_data_loader, loss_function)
        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cwd_parent}/checkpoints/best_model.pth")
            print(f"New best model saved with validation loss: {val_loss:.4f}")

    print("Pre-training completed!")
    return model


def evaluate_model(model, val_data_loader, loss_function):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(val_data_loader)

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            outputs = model(inputs)
            loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


if __name__ == '__main__':
    # 创建检查点目录
    os.makedirs(f"{cwd_parent}/checkpoints", exist_ok=True)

    # 加载数据集和tokenizer
    tokenizer, train_dataset, val_dataset = process_and_load_dataset()

    # 数据加载器
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 根据GPU内存调整
        shuffle=True,
        num_workers=2
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    # 模型配置 (中等规模)
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 1024,  # 增加嵌入维度
        "block_size": 512,  # 增加序列长度
        "num_heads": 16,  # 增加注意力头数
        "num_layers": 12  # 增加层数
    }

    # 初始化模型
    model = MyLLM(**MODEL_CFG)
    model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params / 1e6:.2f}M")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,  # 降低学习率
        weight_decay=0.1  # 权重衰减
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_data_loader) * 10  # 10个epoch
    )

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 训练模型
    trained_model = pretrain(
        num_epochs=10,  # 增加训练轮数
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler
    )

    # 保存最终模型
    torch.save(trained_model.state_dict(), f"{cwd_parent}/checkpoints/final_model.pth")
    print(f"最终模型已保存到 {cwd_parent}/checkpoints/final_model.pth")