# BÁO CÁO SO SÁNH EMBEDDING (Giai đoạn 3)

Đánh giá trên test set 500 query-doc pairs (hold-out, cân bằng 100/ngành, không xuất hiện trong training).

## Bảng tổng kết

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|--------|---------:|---------:|----------:|----:|--------:|
| **hybrid** | 0.6640 | 0.9440 | 0.9900 | 0.7792 | 0.7520 |
| **phobert** | 0.6420 | 0.9640 | 0.9880 | 0.7749 | 0.7472 |
| **e5** | 0.6680 | 0.9640 | 0.9880 | 0.7861 | 0.7614 |
| **gnn_gcn** | 0.0180 | 0.0840 | 0.1580 | 0.0661 | 0.0312 |
| **e5_base_pretrained** | 0.0160 | 0.0540 | 0.0640 | 0.0439 | 0.0119 |
| **gnn_gat** | 0.0260 | 0.0360 | 0.0500 | 0.0437 | 0.0194 |

## Mô tả phương án

- **phobert**: PhoBERT-base-v2 fine-tuned (Vietnamese-only, MultipleNegativesRankingLoss in-batch)
- **e5**: multilingual-E5-base fine-tuned (multilingual, có hard negatives từ negative_sampling)
- **e5_base_pretrained**: multilingual-E5-base PRETRAINED (baseline, không fine-tune)
- **gnn_gcn**: GCN 2-layer trên prereq graph (text features init, PCA bridge cho query)
- **gnn_gat**: GAT 2-layer multi-head trên prereq graph (PCA bridge cho query)
- **hybrid**: Late fusion E5 + GCN (α=0.85 sau khi sweep — xem hybrid_best_alpha.json)

## Phương án tốt nhất & Quyết định

**hybrid** đạt Recall@10 cao nhất (0.9900).

Tuy nhiên hybrid chỉ hơn E5 đơn lẻ 0.20% Recall@10 (0.9900 vs 0.9880), trong khi E5 có MRR & NDCG@10 cao hơn. Để giảm phức tạp pipeline (không cần PCA bridge + concat hybrid vector), **FAISS index được build với E5 fine-tuned**. Hybrid vẫn có thể được kích hoạt sau cho retrieval stage ở Giai đoạn 5 nếu cần đẩy thêm 0.2% Recall.

## Phân tích

- PhoBERT và E5 tương đương ở Recall@10.
- Fine-tune E5 cải thiện Recall@10 +92.40% so với baseline pretrained (0.0640 -> 0.9880).
- Hybrid (E5 + GNN) so với E5 standalone: Δ Recall@10 = +0.20%. GNN bổ sung tín hiệu cấu trúc tiên quyết.
