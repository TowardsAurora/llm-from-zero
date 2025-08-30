import torch
from torch.utils.data import DataLoader
from model import MyLLM
from data_process import process_and_load_dataset,cwd_parent
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pretrain(num_epoches, model=None, train_data_loader=None, optimizer=None, loss_function=None):
    print("start pre-training...")
    model.train()
    for epoch in range(num_epoches):
        loop = tqdm(train_data_loader, total=len(train_data_loader), desc=f'Epoch {epoch + 1}/{num_epoches}')
        total_loss = 0
        for batch in train_data_loader:
            inputs_ids= batch["input_ids"].to(device)
            inputs = inputs_ids[:,:-1]
            labels = inputs_ids[:,1:]

            outputs = model(inputs)
            loss = loss_function(
                outputs.view(-1,outputs.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(),avg_loss=total_loss/(loop.n+1))
    print("finish pre-training...")
    return model


if __name__ == '__main__':
    ## ====Setp1: load tokenized datasets==== ##
    tokenizer ,train_dataset,val_dataset = process_and_load_dataset()
    """  dataset example
    
    # print(train_dataset)
    # print(val_dataset)
    # print(next(iter(val_dataset)))
    
    Dataset({
        features: ['text', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 4
    })
    """
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)


    ## ====Setp2: initialize the model==== ##
    MODEL_CFG = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 512,
        "block_size": 128,
        "num_heads": 8,
        "num_layers": 6
    }
    model = MyLLM(**MODEL_CFG)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #=====Step3: start pre-training=====
    trained_model = pretrain(2, model, train_data_loader, optimizer, loss_function)

    #=====Step4: save the trained model=====
    print(f"model saved at {cwd_parent}/checkpoints/pretrained_llm.pth")
    torch.save(trained_model.state_dict(), f"{cwd_parent}/checkpoints/pretrained_llm.pth")

