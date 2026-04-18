def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text