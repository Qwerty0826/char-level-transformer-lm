import torch
from model.transformer import TransformerLM


def train():
    model = TransformerLM(vocab_size=10000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    dummy_data = torch.randint(0, 10000, (32, 50))

    for step in range(100):
        logits = model(dummy_data)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10000),
            dummy_data.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()
