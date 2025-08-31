import torch
from torch.utils.data import DataLoader
from model import MyLLM
from data_process import process_and_load_dataset, cwd_parent
from tqdm import tqdm
import math
import os
from accelerate import Accelerator, DistributedType
from torch.cuda.amp import autocast

# 初始化 Accelerator
accelerator = Accelerator(
    mixed_precision='fp16',  # 使用混合精度
    gradient_accumulation_steps=4,  # 梯度累积步数
)

device = accelerator.device


def pretrain(num_epochs, model, train_data_loader, val_data_loader, optimizer, loss_function, scheduler=None):
    """预训练函数"""
    print("Starting pre-training with Accelerate...")
    best_val_loss = float('inf')

    # 使用 Accelerator 准备模型、优化器、数据加载器
    model, optimizer, train_data_loader, val_data_loader, scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, val_data_loader, scheduler
    )

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_loop = tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]',
                          disable=not accelerator.is_local_main_process)

        for batch in train_loop:
            with accelerator.accumulate(model):
                input_ids = batch['input_ids'].to(device)
                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]

                # 使用混合精度训练
                with autocast():
                    # 前向传播
                    outputs = model(inputs)
                    loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.reshape(-1))

                # 反向传播（Accelerate 自动处理）
                accelerator.backward(loss)

                # 梯度裁剪
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

                total_loss += accelerator.gather(loss).mean().item()
                avg_loss = total_loss / (train_loop.n + 1)
                train_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        # 验证阶段
        val_loss, perplexity = evaluate_model(model, val_data_loader, loss_function)

        # 只在主进程打印
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), f"{cwd_parent}/checkpoints/best_model.pth")
                print(f"New best model saved with validation loss: {val_loss:.4f}")

    if accelerator.is_local_main_process:
        print("Pre-training completed!")
    return model


def evaluate_model(model, val_data_loader, loss_function):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(val_data_loader)

    for batch in tqdm(val_data_loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        input_ids = batch['input_ids'].to(device)
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        with torch.no_grad():
            # 使用混合精度进行评估
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.reshape(-1))

        total_loss += accelerator.gather(loss).mean().item()

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


if __name__ == '__main__':
    # 创建检查点目录
    if accelerator.is_local_main_process:
        os.makedirs(f"{cwd_parent}/checkpoints", exist_ok=True)

    # 加载数据集和tokenizer
    tokenizer, train_dataset, val_dataset = process_and_load_dataset()

    # 数据加载器
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=8,  # 可以根据GPU数量调整
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 模型配置
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 768,
        "block_size": 256,
        "num_heads": 12,
        "num_layers": 8
    }

    # 初始化模型
    model = MyLLM(**MODEL_CFG)

    # 打印模型参数数量（只在主进程）
    if accelerator.is_local_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params / 1e6:.2f}M")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_data_loader) * 10
    )

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 训练模型
    trained_model = pretrain(
        num_epochs=10,
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler
    )

    # 保存最终模型（只在主进程）
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(trained_model)
        torch.save(unwrapped_model.state_dict(), f"{cwd_parent}/checkpoints/final_model.pth")
        print(f"最终模型已保存到 {cwd_parent}/checkpoints/final_model.pth")