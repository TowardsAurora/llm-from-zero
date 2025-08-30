import torch
from model import MyLLM
from data_process import process_and_load_dataset, cwd_parent
from typing import Optional
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_text(
        prompt: str,
        model: MyLLM,
        tokenizer,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1
) -> str:
    """
    生成文本的优化函数

    参数:
        prompt: 输入提示文本
        model: 训练好的模型
        tokenizer: 分词器
        max_new_tokens: 最大生成token数量
        temperature: 温度参数，控制随机性 (0.1-1.0)
        top_k: 只从概率最高的k个token中采样
        top_p: 只从累积概率达到p的token中采样
        repetition_penalty: 重复惩罚系数

    返回:
        生成的文本
    """
    model.eval()

    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    # 存储已生成的token，用于重复惩罚
    generated_tokens = set(input_ids[0].tolist())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截断输入以适应模型的最大序列长度
            current_input_ids = generated_ids if generated_ids.size(1) <= model.block_size else generated_ids[:,
                                                                                                -model.block_size:]

            # 获取模型输出
            logits = model(current_input_ids)

            # 只关注最后一个token的预测
            logits = logits[:, -1, :] / temperature

            # 应用重复惩罚
            for token_id in generated_tokens:
                if token_id < logits.size(-1):
                    logits[0, token_id] /= repetition_penalty

            # 应用top-k过滤
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                min_val = top_k_values[0, -1].item()
                logits[logits < min_val] = float('-inf')

            # 应用top-p过滤
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # 采样下一个token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 更新已生成的token集合
            generated_tokens.add(next_token_id.item())

            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # 如果遇到结束标记，提前终止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # 解码生成的文本
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def interactive_mode(model, tokenizer):
    """交互式模式"""
    print("进入交互模式，输入 'quit' 退出")

    while True:
        try:
            prompt = input("\n输入提示: ")
            if prompt.lower() == 'quit':
                break

            # 生成文本
            generated_text = generate_text(
                prompt,
                model,
                tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

            print(f"模型生成: {generated_text}")

        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def batch_mode(model, tokenizer, input_file, output_file):
    """批量模式"""
    print(f"批量处理文件: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        results = []
        for prompt in prompts:
            generated_text = generate_text(
                prompt,
                model,
                tokenizer,
                max_new_tokens=100,
                temperature=0.7
            )
            results.append(f"输入: {prompt}\n输出: {generated_text}\n")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(results)

        print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaMA模型推理')
    parser.add_argument('--model_path', type=str, default=f"{cwd_parent}/checkpoints/final_model.pth",
                        help='模型路径')
    parser.add_argument('--interactive', action='store_true',
                        help='进入交互模式')
    parser.add_argument('--input_file', type=str,
                        help='输入文件路径（批量模式）')
    parser.add_argument('--output_file', type=str, default="output.txt",
                        help='输出文件路径（批量模式）')
    parser.add_argument('--prompt', type=str,
                        help='直接输入提示文本')

    args = parser.parse_args()

    # 加载tokenizer
    tokenizer, _, _ = process_and_load_dataset()

    # 模型配置（必须与训练时一致）
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 1024,
        "block_size": 512,
        "num_heads": 16,
        "num_layers": 12
    }

    # 初始化模型
    model = MyLLM(**MODEL_CFG)

    # 加载模型权重
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # 移除可能存在的缓冲区键（如causal_mask）
        keys_to_remove = [k for k in state_dict.keys() if 'causal_mask' in k or 'rotary_emb' in k]
        for k in keys_to_remove:
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print(f"模型已加载: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit(1)

    # 根据参数选择模式
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.input_file:
        batch_mode(model, tokenizer, args.input_file, args.output_file)
    elif args.prompt:
        generated_text = generate_text(
            args.prompt,
            model,
            tokenizer,
            max_new_tokens=100,
            temperature=0.7
        )
        print(f"输入: {args.prompt}")
        print(f"输出: {generated_text}")
    else:
        # 默认交互模式
        interactive_mode(model, tokenizer)