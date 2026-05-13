import os
import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy

# ==========================================================
# Simple Vocabulary Class
# ==========================================================

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        # Special tokens
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        self.stoi = {}
        self.itos = {}

    def build(self, token_lists):
        counter = Counter()

        for tokens in token_lists:
            counter.update(tokens)

        # Add special tokens first
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
            self.itos[idx] = token

        idx = len(self.special_tokens)

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def numericalize(self, tokens):
        unk_idx = self.stoi["<unk>"]
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def __len__(self):
        return len(self.stoi)

# ==========================================================
# Multi30k Dataset
# ==========================================================

class Multi30kDataset(Dataset):
    def __init__(self, split="train", cache_dir="./data", vocab_path="./data/vocabs.pkl"):
        self.split = split
        self.cache_dir = cache_dir
        self.vocab_path = vocab_path

        # Load Hugging Face dataset
        data_dir = os.path.join(cache_dir, "bentrevett_multi30k")

        self.dataset = load_dataset(
            "arrow",
            data_files={
                "train": os.path.join(data_dir, "multi30k-train.arrow"),
                "validation": os.path.join(data_dir, "multi30k-validation.arrow"),
                "test": os.path.join(data_dir, "multi30k-test.arrow"),
            },
            cache_dir=cache_dir,
        )[split]

        # Load spaCy tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # Load or build vocabularies
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                self.src_vocab, self.tgt_vocab = pickle.load(f)
        else:
            self.build_vocab()
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            with open(vocab_path, "wb") as f:
                pickle.dump((self.src_vocab, self.tgt_vocab), f)

        # Convert all sentences to token indices
        self.process_data()

    # ------------------------------------------------------
    def tokenize_de(self, text):
        return [token.text.lower() for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    # ------------------------------------------------------
    def build_vocab(self):
        # Always build vocab from the training split only
        data_dir = os.path.join(self.cache_dir, "bentrevett_multi30k")

        train_data = load_dataset(
            "arrow",
            data_files={
                "train": os.path.join(data_dir, "multi30k-train.arrow")
            },
            cache_dir=self.cache_dir,
        )["train"]

        src_tokens = []
        tgt_tokens = []

        for example in train_data:
            src_tokens.append(self.tokenize_de(example["de"]))
            tgt_tokens.append(self.tokenize_en(example["en"]))

        self.src_vocab = Vocabulary(min_freq=2)
        self.tgt_vocab = Vocabulary(min_freq=2)

        self.src_vocab.build(src_tokens)
        self.tgt_vocab.build(tgt_tokens)

    # ------------------------------------------------------
    def process_data(self):
        self.data = []

        sos_src = self.src_vocab.stoi["<sos>"]
        eos_src = self.src_vocab.stoi["<eos>"]

        sos_tgt = self.tgt_vocab.stoi["<sos>"]
        eos_tgt = self.tgt_vocab.stoi["<eos>"]

        for example in self.dataset:
            # German sentence
            src_tokens = self.tokenize_de(example["de"])
            src_indices = [sos_src]
            src_indices += self.src_vocab.numericalize(src_tokens)
            src_indices += [eos_src]

            # English sentence
            tgt_tokens = self.tokenize_en(example["en"])
            tgt_indices = [sos_tgt]
            tgt_indices += self.tgt_vocab.numericalize(tgt_tokens)
            tgt_indices += [eos_tgt]

            self.data.append(
                (
                    torch.tensor(src_indices, dtype=torch.long),
                    torch.tensor(tgt_indices, dtype=torch.long),
                )
            )

    # ------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------
    def __getitem__(self, idx):
        return self.data[idx]
