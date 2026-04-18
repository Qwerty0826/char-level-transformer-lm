import torch
from model.transformer import TransformerLM
from utils.dataset import load_text


def train():
    # Load real data
    text = load_text("data/tinystories.txt")

    # Simple character encoding (temporary)
    vocab = list(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}

    # Convert text to tokens
    data = torch.tensor([stoi[c] for c in text[:10000]])  # limit for now

    # Create batches
    seq_len = 50
    batch_size = 32

    x = []
    y = []

    for i in range(0, len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+1:i+seq_len+1])

    x = torch.stack(x[:batch_size])
    y = torch.stack(y[:batch_size])

    model = TransformerLM(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(50):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, len(vocab)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()
