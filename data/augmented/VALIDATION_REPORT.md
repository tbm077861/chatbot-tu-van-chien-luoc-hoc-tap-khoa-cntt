# Báo cáo Validation — Giai đoạn 2 (Data Augmentation)

Tổng số sample: **106,958**
Số file nguồn: 15
Lỗi schema: 0
Duplicate ID: 0
Sample hợp lệ ghi vào ALL_samples.jsonl: 106958 (chỉ loại sample có lỗi schema)

## Phân bố theo nguồn
| Source | Số sample | % |
|---|---:|---:|
| cf_svd_augmentation | 60,000 | 56.10% |
| graph_path_sampling | 32,000 | 29.92% |
| negative_sampling | 14,958 | 13.98% |

## Phân bố theo ngành
| Ngành | Số sample | % |
|---|---:|---:|
| CS | 21,395 | 20.00% |
| IT | 21,393 | 20.00% |
| IS | 21,391 | 20.00% |
| SE | 21,391 | 20.00% |
| DS | 21,388 | 20.00% |

## Phân bố ngành × nguồn
| Ngành | Source | Số sample |
|---|---|---:|
| CS | cf_svd_augmentation | 12,000 |
| CS | graph_path_sampling | 6,400 |
| CS | negative_sampling | 2,995 |
| DS | cf_svd_augmentation | 12,000 |
| DS | graph_path_sampling | 6,400 |
| DS | negative_sampling | 2,988 |
| IS | cf_svd_augmentation | 12,000 |
| IS | graph_path_sampling | 6,400 |
| IS | negative_sampling | 2,991 |
| IT | cf_svd_augmentation | 12,000 |
| IT | graph_path_sampling | 6,400 |
| IT | negative_sampling | 2,993 |
| SE | cf_svd_augmentation | 12,000 |
| SE | graph_path_sampling | 6,400 |
| SE | negative_sampling | 2,991 |

## Lỗi schema
Không có lỗi schema nào. ✓

## Kết luận
Đạt mục tiêu **≥85,000 sample** (project_instructions.md mục 10): **106,958**.
