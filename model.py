"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # Get dimension of keys (last dimension)
    d_k = Q.size(-1)

    # Compute raw attention scores: Q K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    # mask == True means "mask out this position"
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Convert scores to probabilities
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)

    # Return both output and attention weights
    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # True where token is PAD
    mask = (src == pad_idx)

    # Reshape to [batch, 1, 1, src_len]
    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.size()
    device = tgt.device

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: upper triangular part above diagonal
    # Shape: [tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1
    )

    # Reshape to [1, 1, tgt_len, tgt_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Combine masks (broadcasting handles batch dimension)
    mask = pad_mask | causal_mask

    return mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout applied to attention weights
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        # --------------------------------------------------
        # 1. Linear projections
        #    [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        # --------------------------------------------------
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # --------------------------------------------------
        # 2. Split into multiple heads
        #    [batch, seq_len, d_model]
        # -> [batch, seq_len, num_heads, d_k]
        # -> [batch, num_heads, seq_len, d_k]
        # --------------------------------------------------
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # --------------------------------------------------
        # 3. Apply scaled dot-product attention
        # --------------------------------------------------
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Optional dropout on attention output
        attn_output = self.dropout(attn_output)

        # --------------------------------------------------
        # 4. Concatenate heads
        #    [batch, num_heads, seq_len, d_k]
        # -> [batch, seq_len, num_heads, d_k]
        # -> [batch, seq_len, d_model]
        # --------------------------------------------------
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # --------------------------------------------------
        # 5. Final linear projection
        # --------------------------------------------------
        output = self.W_o(attn_output)

        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Even indices: sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd indices: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as non-trainable buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        seq_len = x.size(1)

        # Add positional encodings
        x = x + self.pe[:, :seq_len, :]

        # Apply dropout
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Self-attention
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1. Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Cross-attention over encoder memory
        attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        # TODO: Instantiate
        # Save configuration for checkpointing
        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }

        # Store attributes
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)

        # Decoder
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)

        # Final output projection
        self.generator = nn.Linear(d_model, tgt_vocab_size)


        import pickle
        import spacy

        # Load tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # Load vocabularies
        with open("./data/vocabs.pkl", "rb") as f:
            self.src_vocab, self.tgt_vocab = pickle.load(f)
        
        checkpoint_path = "./checkpoints/best_model.pt"

        # Download from Google Drive only if checkpoint is not already present
        if not os.path.exists(checkpoint_path):
            os.makedirs("./checkpoints", exist_ok=True)

            gdown.download(
                id="1qqRwh6a84_qvGNvK50veDCjxVNkFz9Ly",
                output=checkpoint_path,
                quiet=False,
            )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        # Training checkpoints store model weights under "model_state_dict"
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)

        # Special token indices
        self.src_sos_idx = self.src_vocab.stoi["<sos>"]
        self.src_eos_idx = self.src_vocab.stoi["<eos>"]
        self.tgt_sos_idx = self.tgt_vocab.stoi["<sos>"]
        self.tgt_eos_idx = self.tgt_vocab.stoi["<eos>"]

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        # Token embeddings scaled by sqrt(d_model)
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        # Add positional encodings
        x = self.positional_encoding(x)

        # Pass through encoder stack
        memory = self.encoder(x, src_mask)

        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Target embeddings scaled by sqrt(d_model)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encodings
        x = self.positional_encoding(x)

        # Pass through decoder stack
        x = self.decoder(x, memory, src_mask, tgt_mask)

        # Project to target vocabulary logits
        logits = self.generator(x)

        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        memory = self.encode(src, src_mask)

        # Decode target sequence
        logits = self.decode(memory, src_mask, tgt, tgt_mask)

        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        # Local import to avoid circular dependency
        from train import greedy_decode

        # Determine model device
        device = next(self.parameters()).device

        # --------------------------------------------------
        # 1. Tokenize German sentence
        # --------------------------------------------------
        tokens = [
            token.text.lower()
            for token in self.spacy_de.tokenizer(src_sentence)
        ]

        # --------------------------------------------------
        # 2. Convert tokens to indices
        # --------------------------------------------------
        src_indices = [self.src_sos_idx]
        src_indices += self.src_vocab.numericalize(tokens)
        src_indices += [self.src_eos_idx]

        # --------------------------------------------------
        # 3. Convert to tensor
        # --------------------------------------------------
        src = torch.tensor(
            [src_indices],
            dtype=torch.long,
            device=device
        )

        # --------------------------------------------------
        # 4. Create source mask
        # --------------------------------------------------
        src_mask = make_src_mask(src, pad_idx=1)

        # --------------------------------------------------
        # 5. Greedy decoding
        # --------------------------------------------------
        with torch.no_grad():
            pred = greedy_decode(
                self,
                src,
                src_mask,
                max_len=100,
                start_symbol=self.tgt_sos_idx,
                end_symbol=self.tgt_eos_idx,
                device=device,
            )

        # --------------------------------------------------
        # 6. Convert predicted indices to tokens
        # --------------------------------------------------
        pred_indices = pred.squeeze(0).tolist()

        words = []
        for idx in pred_indices:
            if idx in (self.tgt_sos_idx, 1):   # skip <sos> and <pad>
                continue
            if idx == self.tgt_eos_idx:
                break
            words.append(self.tgt_vocab.itos[idx])

        # --------------------------------------------------
        # 7. Detokenize and return
        # --------------------------------------------------
        translation = " ".join(words)

        # Basic cleanup for punctuation spacing
        translation = (
            translation.replace(" ,", ",")
                    .replace(" .", ".")
                    .replace(" !", "!")
                    .replace(" ?", "?")
                    .replace(" ' ", "'")
        )

        return translation.strip()