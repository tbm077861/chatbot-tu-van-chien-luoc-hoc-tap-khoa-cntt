"""Train GCN/GAT trên prerequisite graph và sinh node embedding (Giai đoạn 3).

Kiến trúc:
- Mỗi ngành có 1 graph riêng (NetworkX DiGraph). Hợp nhất 5 graph thành
  1 graph chung với node_id = "{nganh}_{ma_mon}" (trùng với doc_id của corpus).
- Node feature khởi tạo từ text-embedding (PhoBERT đã fine-tune) — embedding
  của text trong corpus. Như vậy GNN tận dụng được signal text + cấu trúc.
- Self-supervised training bằng `link prediction`: cạnh tiên quyết là positive
  edge; sample negative edge ngẫu nhiên. Loss = BCE trên (u·v).
- Output: ma trận embedding [num_nodes × dim], lưu kèm map node_id -> index.

CLI:
    python -m src.embedding.graph_embedder --backbone gcn --epochs 100
    python -m src.embedding.graph_embedder --backbone gat --epochs 100
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import negative_sampling

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_DIR = ROOT / "data" / "processed" / "curriculum_graph"
EMB_DIR = ROOT / "data" / "embeddings"


class GCNEncoder(nn.Module):
    """2-layer GCN cho link prediction."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GATEncoder(nn.Module):
    """2-layer GAT (multi-head)."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads, dropout=dropout)
        # Layer cuối heads=1, concat=False để có out_dim cố định.
        self.conv2 = GATConv(hid_dim * heads, out_dim, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def load_unified_graph() -> tuple[nx.DiGraph, list[str]]:
    """Hợp nhất 5 prereq graph (1/ngành) thành 1 graph với node_id ngành-tag.

    Returns:
        (graph hợp nhất, list node_id theo thứ tự ổn định).
    """
    G = nx.DiGraph()
    for path in sorted(CURRICULUM_DIR.glob("*_prereq_graph.gpickle")):
        nganh = path.stem.split("_")[0]
        with open(path, "rb") as f:
            g_nganh = pickle.load(f)
        # Đổi tên node thành "{nganh}_{ma_mon}".
        # Loại bỏ key trùng để tránh kwarg conflict (data có thể đã có 'nganh').
        for n, data in g_nganh.nodes(data=True):
            clean = {k: v for k, v in data.items() if k not in ("nganh", "ma_mon")}
            G.add_node(f"{nganh}_{n}", nganh=nganh, ma_mon=n, **clean)
        for u, v, data in g_nganh.edges(data=True):
            clean = {k: v for k, v in data.items() if k != "nganh"}
            G.add_edge(f"{nganh}_{u}", f"{nganh}_{v}", nganh=nganh, **clean)
    node_ids = sorted(G.nodes())
    return G, node_ids


def init_features_from_text(
    node_ids: list[str], use_text_embedder: bool, text_model_path: str
) -> torch.Tensor:
    """Khởi tạo node feature.

    - Nếu `use_text_embedder=True`: encode text của corpus bằng model đã fine-tune.
    - Ngược lại: dùng one-hot ngành + so_tc + hk_chuan (8-d vector) làm baseline.
    """
    if use_text_embedder:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(text_model_path)
        # Load corpus.
        id2text: dict[str, str] = {}
        with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                id2text[d["doc_id"]] = d["text"]
        texts = [id2text.get(nid, nid) for nid in node_ids]
        emb = model.encode(
            texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True
        )
        return torch.tensor(emb, dtype=torch.float32)

    # Fallback: feature thủ công 5 (ngành one-hot) + 2 (so_tc/hk) + 1 (loai_bb).
    nganh_list = ["CS", "IS", "DS", "SE", "IT"]
    # Cần thêm metadata từ corpus.
    meta: dict[str, dict] = {}
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["doc_id"]] = d
    feats = []
    for nid in node_ids:
        m = meta.get(nid, {})
        nganh = m.get("nganh", "CS")
        oh = [1.0 if nganh == k else 0.0 for k in nganh_list]
        oh.append(float(m.get("so_tc", 0)) / 5.0)
        oh.append(float(m.get("hk_chuan", 0)) / 9.0)
        oh.append(1.0 if m.get("loai") == "bat_buoc" else 0.0)
        feats.append(oh)
    return torch.tensor(feats, dtype=torch.float32)


def to_pyg_data(G: nx.DiGraph, node_ids: list[str], x: torch.Tensor) -> Data:
    """Convert NetworkX -> PyG Data. Edges: prerequisite (u trước v)."""
    idx = {n: i for i, n in enumerate(node_ids)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    if not edges:
        # Graph rỗng cạnh — tạo self-loops để tránh lỗi.
        edges = [(i, i) for i in range(len(node_ids))]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def train_link_prediction(
    model: nn.Module,
    data: Data,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    """Train self-supervised link prediction.

    Loss = BCE(sigmoid(z_u · z_v), 1 nếu edge thật, 0 nếu negative sampling).
    """
    model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_edge_index = data.edge_index
    num_nodes = data.num_nodes

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model(data.x, pos_edge_index)
        # Positive: edge thật.
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        # Negative: sample N edge giả.
        neg_edge = negative_sampling(
            pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )
        neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=-1)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
        )
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            with torch.no_grad():
                acc = (
                    (torch.sigmoid(scores) > 0.5).float() == labels
                ).float().mean()
            print(f"[ep {ep:03d}] loss={loss.item():.4f} acc={acc.item():.4f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text_init",
        action="store_true",
        help="Dùng PhoBERT fine-tuned làm node feature (mạnh hơn one-hot).",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default=str(EMB_DIR / "phobert_finetuned"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    G, node_ids = load_unified_graph()
    print(
        f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(prereq union từ 5 ngành)"
    )

    x = init_features_from_text(node_ids, args.text_init, args.text_model)
    print(f"[features] x.shape={tuple(x.shape)} (text_init={args.text_init})")

    data = to_pyg_data(G, node_ids, x)
    in_dim = x.size(1)

    if args.backbone == "gcn":
        model = GCNEncoder(in_dim, args.hid, args.out_dim)
    else:
        model = GATEncoder(in_dim, args.hid, args.out_dim, heads=4)

    t0 = time.time()
    model = train_link_prediction(model, data, args.epochs, args.lr, device)
    print(f"[train] done in {time.time() - t0:.1f}s")

    # Encode cuối.
    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()

    out_dir = EMB_DIR / f"gnn_{args.backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "node_embeddings.npy", z)
    with open(out_dir / "node_ids.json", "w", encoding="utf-8") as f:
        json.dump(node_ids, f, ensure_ascii=False, indent=2)
    print(f"[save] embedding [{z.shape[0]} × {z.shape[1]}] -> {out_dir}")


if __name__ == "__main__":
    main()
