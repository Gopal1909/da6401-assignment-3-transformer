"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""
import sys
sys.tracebacklimit = 100
sys.excepthook = lambda t, v, tb: __import__('traceback').print_exception(t, v, tb)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask
from nltk.translate.bleu_score import corpus_bleu

import os
import wandb
from torch.nn.utils.rnn import pad_sequence


from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        # logits: [N, vocab_size]
        # target: [N]

        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                self.smoothing / (self.vocab_size - 2)
            )

            # Put confidence at the true class
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # PAD token gets zero probability everywhere
            true_dist[:, self.pad_idx] = 0

            # Rows where target itself is PAD should be all zeros
            pad_mask = (target == self.pad_idx)
            true_dist[pad_mask] = 0

        loss = self.criterion(log_probs, true_dist)

        # Normalize by number of non-pad tokens
        non_pad = (target != self.pad_idx).sum().clamp(min=1)

        return loss / non_pad


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    total_loss = 0.0
    num_batches = 0

    # Set train/eval mode
    if is_train:
        model.train()
    else:
        model.eval()

    pad_idx = 1  # <pad>

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # input  = <sos> w1 w2 ... wn
        # target = w1 w2 ... wn <eos>
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Create masks
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        # Forward pass
        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten for loss
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = loss_fn(logits, tgt_output)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    # Encode source sentence once
    memory = model.encode(src, src_mask)

    # Start with <sos>
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        # Build target mask for current partial output
        tgt_mask = make_tgt_mask(ys, pad_idx=1)

        # Decode current sequence
        out = model.decode(memory, src_mask, ys, tgt_mask)

        # Get logits for last generated position
        prob = out[:, -1, :]

        # Select token with highest probability
        next_word = torch.argmax(prob, dim=-1).item()

        # Append to output sequence
        next_token = torch.tensor([[next_word]], dtype=torch.long, device=device)
        ys = torch.cat([ys, next_token], dim=1)

        # Stop if <eos> generated
        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    

    model.eval()

    references = []
    hypotheses = []

    sos_idx = 2  # <sos>
    eos_idx = 3  # <eos>
    pad_idx = 1  # <pad>

    def idx_to_token(idx):
        if hasattr(tgt_vocab, "itos"):
            return tgt_vocab.itos[idx]
        return tgt_vocab.lookup_token(idx)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            # Decode one sentence at a time
            for i in range(src.size(0)):
                src_i = src[i:i+1].to(device)
                tgt_i = tgt[i].tolist()

                src_mask = make_src_mask(src_i, pad_idx)

                pred = greedy_decode(
                    model,
                    src_i,
                    src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                pred_tokens = pred.squeeze(0).tolist()

                # Remove special tokens from prediction
                pred_words = []
                for idx in pred_tokens:
                    if idx in (sos_idx, pad_idx):
                        continue
                    if idx == eos_idx:
                        break
                    pred_words.append(idx_to_token(idx))

                # Remove special tokens from reference
                ref_words = []
                for idx in tgt_i:
                    if idx in (sos_idx, pad_idx):
                        continue
                    if idx == eos_idx:
                        break
                    ref_words.append(idx_to_token(idx))

                hypotheses.append(pred_words)
                references.append([ref_words])

    # Convert to 0–100 scale
    bleu = corpus_bleu(references, hypotheses) * 100
    return bleu

# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "model_config": model.config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location="cpu")

    # Restore model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state if provided
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state if provided
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return saved epoch number
    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment


    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)

        src_batch = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=1,   # <pad>
        )

        tgt_batch = pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=1,   # <pad>
        )

        return src_batch, tgt_batch


    # Hyperparameters
    config = {
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 64,
        "num_epochs": 20,
        "warmup_steps": 4000,
        "learning_rate": 1.0,   # base LR for Noam
    }

    wandb.init(
        project="da6401-assignment3",
        config=config,
        settings=wandb.Settings(init_timeout=300)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    train_dataset = Multi30kDataset(split="train")
    val_dataset = Multi30kDataset(split="validation")
    test_dataset = Multi30kDataset(split="test")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model
    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Scheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    # Loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=1,
        smoothing=0.1,
    )

    # Training Loop
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )

        print(
            f"Epoch {epoch+1}/{config['num_epochs']} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            path="checkpoints/best_model.pt",
        )

    # Final BLEU
    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_vocab,
        device=device,
    )

    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()



if __name__ == "__main__":
    print("Started")
    run_training_experiment()
