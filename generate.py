def generate(model, start_token, max_len=50):
    model.eval()
    tokens = [start_token]
    for _ in range(max_len):
        x = torch.tensor(tokens).unsqueeze(0)
        logits = model(x)
        next_token = torch.argmax(logits[0, -1]).item()
        tokens.append(next_token)
    return tokens