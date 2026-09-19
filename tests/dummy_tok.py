"""Whitespace tokenizer with CLS/SEP/MASK for packing tests."""


class DummyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        ids = []
        for tok in text.replace("\n", " ").split():
            ids.append(10 + (sum(ord(c) for c in tok) % 1000))
        return ids
