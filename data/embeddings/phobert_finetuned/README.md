---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:60000
- loss:MultipleNegativesRankingLoss
base_model: vinai/phobert-base-v2
widget:
- source_sentence: 'Em là sinh viên ngành IT, đã hoàn thành học kỳ 6 với GPA tích
    luỹ 7.31 (thang 10). Định hướng nghề: chưa xác định. Em đã học 58 môn. Học kỳ
    7 em nên đăng ký môn nào?'
  sentences:
  - 'Môn Hội Họa (mã 003748) thuộc ngành Công nghệ Thông tin. Số tín chỉ: 3. Học kỳ
    chuẩn: 7. Loại: tự chọn.'
  - 'Môn Anh văn 4 (mã 015258) thuộc ngành Khoa học Máy tính. Số tín chỉ: 3. Học kỳ
    chuẩn: 5. Loại: bắt buộc.'
  - 'Môn Toán cao cấp 2 (mã 003595) thuộc ngành Kỹ thuật Phần mềm. Số tín chỉ: 2.
    Học kỳ chuẩn: 3. Loại: bắt buộc.'
- source_sentence: Em là sinh viên ngành DS, đã học 9 môn với GPA tích luỹ 7.54. Em
    vừa hoàn thành tới HK4. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK6?
  sentences:
  - 'Môn Tiếp thị điện tử (mã 003453) thuộc ngành Khoa học Dữ liệu. Số tín chỉ: 3.
    Học kỳ chuẩn: 6. Loại: tự chọn.'
  - 'Môn Những vấn đề xã hội và nghề nghiệp (mã 002215) thuộc ngành Công nghệ Thông
    tin. Số tín chỉ: 3. Học kỳ chuẩn: 6. Loại: bắt buộc.'
  - 'Môn Phân tích chuỗi thời gian (mã 014110) thuộc ngành Khoa học Dữ liệu. Số tín
    chỉ: 3. Học kỳ chuẩn: 5. Loại: tự chọn.'
- source_sentence: Em là sinh viên ngành CS, đã học 12 môn với GPA tích luỹ 7.19.
    Em vừa hoàn thành tới HK7. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK8?
  sentences:
  - 'Môn Hội Họa (mã 003748) thuộc ngành Kỹ thuật Phần mềm. Số tín chỉ: 3. Học kỳ
    chuẩn: 7. Loại: tự chọn.'
  - 'Môn Chủ nghĩa xã hội khoa học (mã 013803) thuộc ngành Kỹ thuật Phần mềm. Số tín
    chỉ: 2. Học kỳ chuẩn: 5. Loại: bắt buộc. Điều kiện học trước: Triết học Mác -
    Lênin (013801), Kinh tế chính trị Mác-Lênin (013802).'
  - 'Môn Phát triển giao diện ứng dung (mã 015436) thuộc ngành Khoa học Máy tính.
    Số tín chỉ: 3. Học kỳ chuẩn: 8. Loại: tự chọn. Điều kiện học trước: Hệ Thống và
    Công nghệ Web (002399).'
- source_sentence: Em là sinh viên ngành CS, đã học 4 môn với GPA tích luỹ 7.52. Em
    vừa hoàn thành tới HK6. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK7?
  sentences:
  - 'Môn Tiếng Việt thực hành (mã 003633) thuộc ngành Công nghệ Thông tin. Số tín
    chỉ: 3. Học kỳ chuẩn: 7. Loại: tự chọn.'
  - 'Môn Trí tuệ nhân tạo (mã 001954) thuộc ngành Khoa học Máy tính. Số tín chỉ: 3.
    Học kỳ chuẩn: 4. Loại: bắt buộc. Điều kiện học trước: Cấu trúc rời rạc (001508).'
  - 'Môn Kỹ năng sử dụng bàn phím và thiết bị văn phòng (mã 014192) thuộc ngành Khoa
    học Máy tính. Số tín chỉ: 3. Học kỳ chuẩn: 7. Loại: tự chọn.'
- source_sentence: Em là sinh viên ngành SE, đã học 4 môn với GPA tích luỹ 7.78. Em
    vừa hoàn thành tới HK3. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK7?
  sentences:
  - 'Môn Lý thuyết trò chơi (mã 014117) thuộc ngành Khoa học Dữ liệu. Số tín chỉ:
    3. Học kỳ chuẩn: 7. Loại: tự chọn.'
  - 'Môn Quản lý dự án hệ thống thông tin (mã 015013) thuộc ngành Công nghệ Thông
    tin. Số tín chỉ: 3. Học kỳ chuẩn: 6. Loại: tự chọn.'
  - 'Môn Hệ quản trị  cơ sở dữ liệu (mã 001724) thuộc ngành Kỹ thuật Phần mềm. Số
    tín chỉ: 3. Học kỳ chuẩn: 4. Loại: tự chọn. Điều kiện học trước: Hệ cơ sở dữ liệu
    (001922).'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on vinai/phobert-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2) <!-- at revision e2375d266bdf39c6e8e9a87af16a5da3190b0cc8 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'RobertaModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Em là sinh viên ngành SE, đã học 4 môn với GPA tích luỹ 7.78. Em vừa hoàn thành tới HK3. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK7?',
    'Môn Hệ quản trị  cơ sở dữ liệu (mã 001724) thuộc ngành Kỹ thuật Phần mềm. Số tín chỉ: 3. Học kỳ chuẩn: 4. Loại: tự chọn. Điều kiện học trước: Hệ cơ sở dữ liệu (001922).',
    'Môn Lý thuyết trò chơi (mã 014117) thuộc ngành Khoa học Dữ liệu. Số tín chỉ: 3. Học kỳ chuẩn: 7. Loại: tự chọn.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.7250,  0.1346],
#         [ 0.7250,  1.0000, -0.0683],
#         [ 0.1346, -0.0683,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 60,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 44 tokens</li><li>mean: 48.56 tokens</li><li>max: 53 tokens</li></ul> | <ul><li>min: 36 tokens</li><li>mean: 46.95 tokens</li><li>max: 82 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                         | sentence_1                                                                                                                                                                                         |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Em là sinh viên ngành IT, đã học 2 môn với GPA tích luỹ 8.15. Em vừa hoàn thành tới HK2. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK4?</code>                       | <code>Môn Kiến trúc và cài đặt hệ quản trị CSDL (mã 001843) thuộc ngành Công nghệ Thông tin. Số tín chỉ: 3. Học kỳ chuẩn: 4. Loại: tự chọn. Điều kiện học trước: Hệ cơ sở dữ liệu (001922).</code> |
  | <code>Em là sinh viên ngành DS, đã hoàn thành học kỳ 2 với GPA tích luỹ 7.35 (thang 10). Định hướng nghề: chưa xác định. Em đã học 18 môn. Học kỳ 3 em nên đăng ký môn nào?</code> | <code>Môn Tiếng Anh 2 (mã 015254) thuộc ngành Khoa học Dữ liệu. Số tín chỉ: 3. Học kỳ chuẩn: 3. Loại: bắt buộc. Không tính GPA.</code>                                                             |
  | <code>Em là sinh viên ngành CS, đã học 8 môn với GPA tích luỹ 7.57. Em vừa hoàn thành tới HK7. Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK8?</code>                       | <code>Môn Đồ họa máy tính (mã 015033) thuộc ngành Khoa học Máy tính. Số tín chỉ: 3. Học kỳ chuẩn: 8. Loại: tự chọn.</code>                                                                         |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `fp16`: True
- `disable_tqdm`: False
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2667 | 500  | 1.3729        |
| 0.5333 | 1000 | 0.7754        |
| 0.8    | 1500 | 0.7599        |


### Training Time
- **Training**: 4.7 minutes

### Framework Versions
- Python: 3.10.11
- Sentence Transformers: 5.4.1
- Transformers: 5.5.0
- PyTorch: 2.12.0.dev20260408+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->