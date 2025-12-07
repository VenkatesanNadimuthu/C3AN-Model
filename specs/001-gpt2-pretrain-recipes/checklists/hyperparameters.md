# Hyperparameters Checklist: GPT-2 Pre-training on Recipe Dataset

**Purpose**: Track planned hyperparameters for tokenizer training and model pre-training
**Created**: 2025-12-06
**Feature**: [spec.md](../spec.md)

## Tokenizer Hyperparameters

- [x] HP001 - **vocab_size**: 12,000 tokens (optimized for recipe corpus) [Spec §FR-004] ✓ CONFIRMED
- [x] HP002 - **min_frequency**: 2 [Spec §FR-004, plan.md §TOKENIZER_CONFIG] ✓ CONFIRMED
- [x] HP003 - **special_tokens**: [BOS], [EOS], [UNK], [PAD] — 4 reserved tokens [Spec §FR-005] ✓ CONFIRMED

## Model Architecture (GPT2Config) — ENHANCED: GPT-2 Small

- [x] HP004 - **n_positions**: Maximum sequence length — 1024 tokens [Spec §FR-007, Key Entities] ✓ UPDATED
- [x] HP005 - **n_embd**: 768 embedding dimension (GPT-2 Small) [Spec §FR-008] ✓ UPDATED
- [x] HP006 - **n_layer**: 12 transformer layers (GPT-2 Small) [Spec §FR-008] ✓ UPDATED
- [x] HP007 - **n_head**: 12 attention heads (GPT-2 Small, divides 768 evenly) [Spec §FR-008] ✓ UPDATED
- [x] HP008 - **vocab_size**: 12,000 (must match tokenizer) [Spec §FR-008] ✓ CONFIRMED
- [x] HP009 - **bos_token_id**: Set from tokenizer at runtime (ID: 2) [Spec §FR-008] ✓ CONFIRMED
- [x] HP010 - **eos_token_id**: Set from tokenizer at runtime (ID: 3) [Spec §FR-008] ✓ CONFIRMED
- [x] HP011 - **pad_token_id**: Set from tokenizer at runtime (ID: 0) [Spec §FR-008] ✓ CONFIRMED

## Training Hyperparameters (TrainingArguments) — ENHANCED

- [x] HP012 - **num_train_epochs**: 15 epochs (INCREASED for better convergence) [Spec §SC-002, User Story 2] ✓ UPDATED
- [x] HP013 - **per_device_train_batch_size**: 4 (A100 40GB allows larger batches) [Spec §FR-010] ✓ CONFIRMED
- [x] HP014 - **gradient_accumulation_steps**: 8 (effective batch size = 4 × 8 = 32) [Spec §FR-010] ✓ UPDATED
- [x] HP015 - **learning_rate**: 3e-5 (LOWERED for stability) [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ UPDATED
- [x] HP016 - **weight_decay**: 0.01 [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP017 - **warmup_steps**: 1500 (INCREASED for larger model) [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ UPDATED
- [x] HP018 - **fp16**: True — mixed precision training [Spec §FR-009, SC-002] ✓ CONFIRMED
- [x] HP019 - **logging_steps**: 100 [User Story 2, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP020 - **save_strategy**: "epoch" — save checkpoint every epoch [Spec §FR-011] ✓ CONFIRMED
- [x] HP021 - **save_total_limit**: 5 checkpoints (keep last 5) [Spec §FR-011] ✓ UPDATED
- [x] HP022a - **max_grad_norm**: 1.0 (NEW: gradient clipping) [plan.md §TRAINING_CONFIG] ✓ NEW
- [x] HP022b - **lr_scheduler_type**: "cosine" (NEW: cosine annealing) [plan.md §TRAINING_CONFIG] ✓ NEW
- [x] HP022c - **label_smoothing_factor**: 0.1 (NEW: reduce overfitting) [plan.md §TRAINING_CONFIG] ✓ NEW

## Inference Hyperparameters (model.generate)

- [x] HP022 - **max_length**: Not used; using max_new_tokens instead [plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP023 - **max_new_tokens**: 200 [Spec §SC-004, plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP024 - **temperature**: 0.8 [Spec §FR-012, plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP025 - **top_k**: 50 [Spec §FR-012, plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP026 - **top_p**: 0.92 [Spec §FR-012, plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP027 - **do_sample**: True [Spec §FR-012, plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP028 - **repetition_penalty**: 1.1 [plan.md §INFERENCE_CONFIG] ✓ CONFIRMED
- [x] HP029 - **eos_token_id**: Set from tokenizer at runtime [Spec §FR-012, User Story 3] ✓ CONFIRMED

## Memory Optimization

- [x] HP030 - **gradient_checkpointing**: False (not needed with A100 40GB) [Spec §FR-010] ✓ CONFIRMED
- [x] HP031 - **dataloader_num_workers**: 2 (for Colab) [Spec §FR-007] ✓ CONFIRMED

## Fine-tuning Hyperparameters (Phase 2) — ENHANCED

- [x] HP032 - **num_train_epochs**: 8 epochs (INCREASED for instruction learning) [Spec §SC-006] ✓ UPDATED
- [x] HP033 - **per_device_train_batch_size**: 2 (smaller for instruction data) [plan.md §FINETUNE_CONFIG] ✓ CONFIRMED
- [x] HP034 - **gradient_accumulation_steps**: 16 (effective batch = 32) [plan.md §FINETUNE_CONFIG] ✓ UPDATED
- [x] HP035 - **learning_rate**: 5e-6 (LOWERED: preserve pre-trained knowledge) [Spec §FR-017] ✓ UPDATED
- [x] HP036 - **warmup_ratio**: 0.15 (INCREASED: 15% warmup) [plan.md §FINETUNE_CONFIG] ✓ UPDATED
- [x] HP037 - **max_grad_norm**: 1.0 (gradient clipping) [plan.md §FINETUNE_CONFIG] ✓ NEW
- [x] HP038 - **lr_scheduler_type**: "cosine" (cosine annealing) [plan.md §FINETUNE_CONFIG] ✓ NEW
- [x] HP039 - **label_smoothing_factor**: 0.1 (reduce overfitting) [plan.md §FINETUNE_CONFIG] ✓ NEW
- [x] HP040 - **max_seq_length**: 1024 (match model n_positions) [plan.md §FINETUNE_CONFIG] ✓ CONFIRMED

## Notes

- ✓ UPDATED values reflect enhanced GPT-2 Small configuration (December 2024)
- ✓ CONFIRMED values remain unchanged from original spec
- ✓ NEW values are additions to improve training quality
- GPT-2 Small (12 layers, 768 embed, 12 heads, ~125M params) selected for coherent recipe generation
- Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 32
- Cosine learning rate scheduling for smoother convergence
- Gradient clipping (max_grad_norm=1.0) for training stability
- Label smoothing (0.1) to reduce overfitting on small dataset
- Data file uploaded directly to Colab runtime (not Google Drive)
- Recipe format: **Structured format** with [BOS]/[EOS] delimiters and explicit field markers (`**Title:**`, `**Ingredients:**` as bulleted list, `**Instructions:**` as numbered list)
