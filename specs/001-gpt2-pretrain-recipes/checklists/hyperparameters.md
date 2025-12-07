# Hyperparameters Checklist: GPT-2 Pre-training on Recipe Dataset

**Purpose**: Track planned hyperparameters for tokenizer training and model pre-training
**Created**: 2025-12-06
**Feature**: [spec.md](../spec.md)

## Tokenizer Hyperparameters

- [x] HP001 - **vocab_size**: 12,000 tokens (optimized for recipe corpus) [Spec §FR-004] ✓ CONFIRMED
- [x] HP002 - **min_frequency**: 2 [Spec §FR-004, plan.md §TOKENIZER_CONFIG] ✓ CONFIRMED
- [x] HP003 - **special_tokens**: [BOS], [EOS], [UNK], [PAD] — 4 reserved tokens [Spec §FR-005] ✓ CONFIRMED

## Model Architecture (GPT2Config)

- [x] HP004 - **n_positions**: Maximum sequence length — 3000 tokens [Spec §FR-007, Key Entities] ✓ CONFIRMED
- [x] HP005 - **n_embd**: 512 embedding dimension (GPT-2 Mini) [Spec §FR-008] ✓ CONFIRMED
- [x] HP006 - **n_layer**: 6 transformer layers (GPT-2 Mini) [Spec §FR-008] ✓ CONFIRMED
- [x] HP007 - **n_head**: 8 attention heads (GPT-2 Mini, divides 512 evenly) [Spec §FR-008] ✓ CONFIRMED
- [x] HP008 - **vocab_size**: 12,000 (must match tokenizer) [Spec §FR-008] ✓ CONFIRMED
- [x] HP009 - **bos_token_id**: Set from tokenizer at runtime (ID: 2) [Spec §FR-008] ✓ CONFIRMED
- [x] HP010 - **eos_token_id**: Set from tokenizer at runtime (ID: 3) [Spec §FR-008] ✓ CONFIRMED
- [x] HP011 - **pad_token_id**: Set from tokenizer at runtime (ID: 0) [Spec §FR-008] ✓ CONFIRMED

## Training Hyperparameters (TrainingArguments)

- [x] HP012 - **num_train_epochs**: 7 epochs (optimized for 8.6K samples) [Spec §SC-002, User Story 2] ✓ CONFIRMED
- [x] HP013 - **per_device_train_batch_size**: 4 (A100 40GB allows larger batches) [Spec §FR-010] ✓ CONFIRMED
- [x] HP014 - **gradient_accumulation_steps**: 4 (effective batch size = 4 × 4 = 16) [Spec §FR-010] ✓ CONFIRMED
- [x] HP015 - **learning_rate**: 5e-5 [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP016 - **weight_decay**: 0.01 [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP017 - **warmup_steps**: 500 [Spec §FR-009, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP018 - **fp16**: True — mixed precision training [Spec §FR-009, SC-002] ✓ CONFIRMED
- [x] HP019 - **logging_steps**: 100 [User Story 2, plan.md §TRAINING_CONFIG] ✓ CONFIRMED
- [x] HP020 - **save_strategy**: "epoch" — save checkpoint every epoch [Spec §FR-011] ✓ CONFIRMED
- [x] HP021 - **save_total_limit**: 7 checkpoints (one per epoch) [Spec §FR-011] ✓ CONFIRMED

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
- [ ] HP031 - **dataloader_num_workers**: Number of data loading workers (suggested: 0–2 for Colab) [Spec §FR-007]

## Notes

- ✓ CONFIRMED values are finalized from clarification session
- Hyperparameters marked "suggested" require tuning based on actual GPU memory and training dynamics
- Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
- For A100 GPU (≈40 GB), batch size of 4 with gradient_accumulation_steps=4 achieves effective batch size of 16
- GPT-2 Mini (6 layers, 512 embed, 8 heads, ~50M params) selected for training efficiency
- Data file uploaded directly to Colab runtime (not Google Drive)
- Recipe format: **Structured format** with [BOS]/[EOS] delimiters and explicit field markers (`**Title:**`, `**Ingredients:**` as bulleted list, `**Instructions:**` as numbered list)
- Space-padding no longer required—structured format uses attention masking for variable-length sequences
