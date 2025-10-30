import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.tokenizer import CharTokenizer
from utils.dataset import TextDataset
from models.hybrid_gpt import HybridGPT

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = CharTokenizer(["TinyStories training sample"])
    dataset = TextDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HybridGPT(vocab_size=len(tokenizer.chars)).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4)

    for epoch in range(3):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "results/checkpoints/hybrid_gpt.pt")
    print("Model saved to results/checkpoints/hybrid_gpt.pt")

if __name__ == "__main__":
    train_model()
