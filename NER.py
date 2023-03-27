import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# 自定义数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # 对句子进行编码
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 将标签添加到编码中
        label_encoding = [0] + label + [0] * (self.max_length - len(label) - 1)
        label_tensor = torch.tensor(label_encoding, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }

# 定义BERT NER模型
class BERT_NER(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERT_NER, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

# 超参数
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_LABELS = 3  # 根据具体任务调整
BERT_MODEL_NAME = 'bert-base-multilingual-cased'

# 数据预处理：将输入文本和标签转换为适当的格式
# sentences: 列表，包含所有句子
# labels: 列表，包含所有句子的标签
# 示例数据
sentences = [
    "John lives in New York.",
    "玛丽在北京工作。",
    "Alice visited the Great Wall last summer.",
    "李华是一位著名的歌手。"
]

labels = [
    [1, 0, 0, 0, 2, 2, 2, 0],
    [1, 0, 2, 0, 0, 0, 0],
    [1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 实体标签映射
label_map = {
    "O": 0,
    "B-PER": 1,
    "B-LOC": 2
}

sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences, labels, test_size=0.5)

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BERT_NER(BERT_MODEL_NAME, NUM_LABELS)

# 创建数据加载器
train_dataset = NERDataset(sentences_train, labels_train, tokenizer, MAX_LENGTH)
val_dataset = NERDataset(sentences_val, labels_val, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器和学习率调整策略
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练函数
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, NUM_LABELS), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, NUM_LABELS), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 训练和验证模型
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    print('-' * 10)
    train(model, train_loader, optimizer, scheduler, device)
    val_loss = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}\n")
print("Training complete.")

