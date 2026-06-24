import torch
from .model import SASRec
from .database import load_user_history, load_all_products


def search_products(keyword: str, products: dict) -> list:
    kws = keyword.strip().lower().split()
    if not kws:
        return list(products.keys())
    scored = []
    for pid, p in products.items():
        text = f"{p['name']} {p['tags']} {p['category']}".lower()
        hits = sum(1 for kw in kws if kw in text)
        if hits == len(kws):
            name_hits = sum(1 for kw in kws if kw in p['name'].lower())
            scored.append((pid, name_hits * 2 + hits))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored]


@torch.no_grad()
def get_recommendations(model: SASRec, user_id: int,
                        keyword: str = "", top_k: int = 10) -> list:
    model.eval()
    products = load_all_products()
    history = load_user_history(user_id, max_len=20)

    if keyword.strip():
        candidate_ids = search_products(keyword, products)
        if not candidate_ids:
            return []
        mode = "search"
    else:
        seen = set(history)
        candidate_ids = [pid for pid in products if pid not in seen]
        mode = "recommend"

    if not candidate_ids:
        return []

    if history:
        max_len = 20
        seq = history[-max_len:]
        real_len = len(seq)
        pad = seq + [0] * (max_len - real_len)
        seq_t = torch.tensor([pad], dtype=torch.long)
        len_t = torch.tensor([real_len], dtype=torch.long)
        scores = torch.softmax(model.predict_scores(model(seq_t, len_t))[0], dim=0)

        if mode == "search":
            scored = []
            for rank, pid in enumerate(candidate_ids):
                relevance = (len(candidate_ids) - rank) / len(candidate_ids)
                transformer_score = float(scores[pid]) if pid < len(scores) else 0.0
                scored.append((pid, transformer_score * 0.6 + relevance * 0.4))
        else:
            scored = [(pid, float(scores[pid])) for pid in candidate_ids
                      if pid < len(scores)]
    else:
        scored = [(pid, products[pid]["rating"]) for pid in candidate_ids]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        {**products[pid], "id": pid, "score": round(score, 4)}
        for pid, score in scored[:top_k]
    ]
