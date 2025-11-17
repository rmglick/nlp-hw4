import os
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)

    
    def process_data(self, data_folder, split, tokenizer):
        data = []
    
        # Load input questions
        input_file = os.path.join(data_folder, f"{split}.nl")
        with open(input_file, 'r') as f:
            questions = [line.strip() for line in f.readlines()]
    
        if split in ["train", "dev"]:
            sql_file = os.path.join(data_folder, f"{split}.sql")
            with open(sql_file, 'r') as f:
                sqls = [line.strip() for line in f.readlines()]
        
            for question, sql in zip(questions, sqls):
                input_ids = tokenizer.encode(question, max_length=512, truncation=True)
                attention_mask = [1] * len(input_ids)
                label_ids = tokenizer.encode(sql, max_length=256, truncation=True)
            
                data.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long)
                })
        else:
            for question in questions:
                input_ids = tokenizer.encode(question, max_length=512, truncation=True)
                attention_mask = [1] * len(input_ids)
            
                data.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
                })
    
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''Collate function for train/dev with proper tensor types.'''
    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_label_len = max(len(item["labels"]) for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        pad_input = (max_input_len - len(item["input_ids"]))
        input_ids = torch.cat([
            item["input_ids"], 
            torch.zeros(pad_input, dtype=torch.long)
        ])
        attn_mask = torch.cat([
            item["attention_mask"], 
            torch.zeros(pad_input, dtype=torch.long)
        ])
        
        pad_labels = (max_label_len - len(item["labels"]))
        labels = torch.cat([
            item["labels"], 
            torch.full((pad_labels,), -100, dtype=torch.long)
        ])
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attn_mask)
        labels_list.append(labels)
    
    return (
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list)
    )

def test_collate_fn(batch):
    '''Collate function for test set (no labels).'''
    max_input_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    
    for item in batch:
        pad_input = (max_input_len - len(item["input_ids"]))
        input_ids = torch.cat([
            item["input_ids"], 
            torch.zeros(pad_input, dtype=torch.long)
        ])
        attn_mask = torch.cat([
            item["attention_mask"], 
            torch.zeros(pad_input, dtype=torch.long)
        ])
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attn_mask)
    
    return (
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list)
    )

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = (split == "train")
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader

def load_prompting_data(data_folder):
    def load_lines(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
