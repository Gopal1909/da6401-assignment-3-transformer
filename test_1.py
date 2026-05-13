import pickle
from model import Transformer

# Load vocabularies
with open("./data/vocabs.pkl", "rb") as f:
    src_vocab, tgt_vocab = pickle.load(f)

# Create model with correct vocabulary sizes
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
)

model.eval()

sentence = "Ein kleines Mädchen spielt im Park."
print(model.infer(sentence))