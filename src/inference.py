from typing import Optional
import torch
from model import MyLLM
from data_process import process_and_load_dataset,cwd_parent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def generate_text(prompt, model, tokenizer, max_new_tokens=50):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     generated_ids = input_ids.clone()
#
#     for _ in range(max_new_tokens):
#         current_input_ids = generated_ids if generated_ids.size(1) <= model.block_size else generated_ids[:, -model.block_size:]
#
#         with torch.no_grad():
#             logits = model(current_input_ids)
#
#         logits_last = logits[:, -1, :]
#         next_token_id = torch.argmax(logits_last, dim=-1).unsqueeze(0)
#
#         generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
#
#         if next_token_id.item() == tokenizer.eos_token_id:
#             break
#
#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_text(
        prompt: str,
        model: MyLLM,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
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
    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    # 存储已生成的token，用于重复惩罚
    generated_tokens = set()

    for _ in range(max_new_tokens):
        # 截断输入以适应模型的最大序列长度
        current_input_ids = generated_ids if generated_ids.size(1) <= model.block_size else generated_ids[:,
                                                                                            -model.block_size:]

        with torch.no_grad():
            logits = model(current_input_ids)

        # 应用温度参数
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

if __name__ == '__main__':
    tokenizer,_, _ = process_and_load_dataset()
    ## ====Setp2: initialize the model==== ##
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 512,
        "block_size": 128,
        "num_heads": 8,
        "num_layers": 6
    }
    model = MyLLM(**MODEL_CFG)
    # state_dict = torch.load(f"{cwd_parent}/checkpoints/pretrained_llm.pth", map_location=device)
    state_dict = torch.load(f"{cwd_parent}/checkpoints/instruction_turning_llm.pth", map_location=device)
    keys_to_remove = [k for k in state_dict.keys() if 'tril' in k]
    for k in keys_to_remove:
        del state_dict[k]

    # 加载状态字典，允许不严格匹配
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    prompt_text = "韩立"
    generated_text = generate_text(prompt_text, model, tokenizer)

    print(f"输入: {prompt_text}")
    print(f"输出: {generated_text}")