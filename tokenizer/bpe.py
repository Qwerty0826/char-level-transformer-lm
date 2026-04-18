class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, text):
        # byte-level encoding
        tokens = list(text.encode("utf-8"))
        # TODO: implement merge logic
        return self.vocab, self.merges

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")
