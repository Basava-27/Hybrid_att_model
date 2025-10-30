import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, split="train", block_size=128):
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        texts = [x["text"] for x in dataset]
        self.tokenizer = tokenizer
        self.data = tokenizer.encode("".join(texts))
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
