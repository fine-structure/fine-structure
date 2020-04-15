

class CharTokenizer():

    pad_token = '<pad>'
    bos_token = '<bos>'
    eos_token = '<eos>'
    oov_token = '<unk>'

    def __init__(self):

        self.token2index = {}
        self.index2token = {}
        self.token_counts = {}

    def build_vocab(self, texts):

        for text in texts:
            for char in text:
                if char in self.token_counts:
                    self.token_counts[char] += 1
                else:
                    self.token_counts[char] = 1

        token_counts = list(self.token_counts.items())
        token_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_vocab = [self.pad_token, self.bos_token, self.eos_token, self.oov_token]
        sorted_vocab.extend(token_count[0] for token_count in token_counts)

        self.token2index = dict(zip(sorted_vocab, list(range(len(sorted_vocab)))))
        self.index2token = {index: token for token, index in self.token2index.items()}
        self.vocab_size = len(self.token2index)

    def encode(self, texts, add_special_tokens=True):

        if isinstance(texts, str):
            tokens = self.tokenize(texts, add_special_tokens)
            encoded = [self.encode_token(token) for token in tokens]
        else:
            encoded = []
            for text in texts:
                tokens = self.tokenize(text, add_special_tokens)
                encoded.append([self.encode_token(token) for token in tokens])

        return encoded

    def encode_token(self, token):
        return self.token2index.get(token, self.token2index[self.oov_token])

    def tokenize(self, text, add_special_tokens=True):

        if add_special_tokens:
            tokens = [self.bos_token] + list(text) + [self.eos_token]
        else:
            tokens = list(text)

        return tokens

    def decode(self, token_indices):

        tokens = [self.index2token[idx] for idx in token_indices]
        return ''.join(tokens)