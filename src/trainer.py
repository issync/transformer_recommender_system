import random
import torch
import torch.nn.functional as F
from .model import SASRec
from .database import load_user_history, get_conn


def build_training_data(max_len: int = 20) -> list:
    conn = get_conn()
    user_ids = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM behaviors").fetchall()]
    conn.close()
    samples = []
    for uid in user_ids:
        hist = load_user_history(uid, max_len)
        for t in range(1, len(hist)):
            samples.append((hist[:t], hist[t]))
    return samples


def collate_fn(batch: list, max_len: int = 20):
    seqs, targets = zip(*batch)
    padded, lengths = [], []
    for seq in seqs:
        seq = seq[-max_len:]
        lengths.append(len(seq))
        padded.append(seq + [0] * (max_len - len(seq)))
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def train(num_items: int, epochs: int = 150, lr: float = 1e-3,
          batch_size: int = 16, d_model: int = 64) -> SASRec:
    samples = build_training_data()
    model = SASRec(num_items=num_items, d_model=d_model, num_heads=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        total_loss, n = 0.0, 0
        for i in range(0, len(samples), batch_size):
            seq_t, tgt_t, len_t = collate_fn(samples[i:i + batch_size])
            user_repr = model(seq_t, len_t)
            loss = F.cross_entropy(model.predict_scores(user_repr), tgt_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  Loss: {total_loss/max(n,1):.4f}")

    return model
