import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data_deal.dataload import SST2Dataset, read_tsv
from model.bert import get_model
from train.train import train_bert, evaluate_bert, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'bert-base-uncased-local'
BATCH_SIZE = 32
MAX_LEN = 128
EPOCHS = 3
LOG_FILE = 'bert'

def main():
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


    train_data = read_tsv('./data/train.tsv')
    val_data = read_tsv('./data/dev.tsv')
    test_data = read_tsv('./data/test.tsv',test=True)


    train_set = SST2Dataset(train_data, tokenizer)
    val_set = SST2Dataset(val_data, tokenizer)
    test_set = SST2Dataset(test_data, tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    model = get_model(MODEL_NAME, num_labels=2).to(DEVICE)
    train_bert(model, train_loader, val_loader, DEVICE,log_file=LOG_FILE, epochs=EPOCHS,lr=2e-5)
    # 加载最佳模型
    model.load_state_dict(torch.load(f"{LOG_FILE}.pt", map_location=DEVICE))
    acc, f1 = evaluate_bert(model, test_loader, DEVICE)
    test_str = f"Test ACC={acc:.4f}, F1={f1:.4f}\n"
    print(test_str.strip())
    # 追加写入测试结果
    with open(f"./result/{LOG_FILE}.txt", 'a', encoding='utf-8') as f_log:
        f_log.write(test_str)

if __name__ == "__main__":
    main()
