# Implementation Plan: GPT-2 Pre-training on Recipe Dataset

**Branch**: `001-gpt2-pretrain-recipes` | **Date**: 2025-12-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-gpt2-pretrain-recipes/spec.md`
**Author**: AI Research Engineer

---

## Summary

Build and pre-train a GPT-2 Mini language model (~50M parameters) from scratch on a domain-specific recipe dataset containing 8,500+ entries in **structured format** (with explicit `**Title:**`, `**Ingredients:**`, `**Instructions:**` field markers). The implementation trains a custom Byte Pair Encoding tokenizer, initializes the transformer architecture with random weights, and executes 7 epochs of causal language modeling using Hugging Face Trainer with FP16 mixed precision. The entire pipeline is optimized for Google Colab's A100 GPU (≈40 GB VRAM) and produces a checkpoint capable of generating **structured recipe text** with distinct labeled sections from prompts.

---

## Technical Context

**Language/Version**: Python 3.10+ (Colab default)  
**Primary Dependencies**: PyTorch 2.x, transformers 4.x, tokenizers 0.x  
**Storage**: Local filesystem (Colab runtime) for checkpoints and tokenizer artifacts  
**Testing**: Not permitted by constitution (no automated or manual test tasks)  
**Target Platform**: Google Colab with NVIDIA A100 GPU  
**Project Type**: Single Jupyter Notebook (Colab-native)  
**Performance Goals**: Complete 7 epochs on 8,500 recipes in <2 hours; generate coherent 50+ token continuations  
**Constraints**: ≈40 GB GPU memory; 3000-token max sequence length; FP16 required  
**Scale/Scope**: ~50M parameter model; 12K vocabulary; 8,500 training samples

---

## Constitution Check

*GATE: ✅ PASSED — All principles satisfied*

| Principle | Status | Notes |
|-----------|--------|-------|
| Clean Code Discipline | ✅ | Each notebook section has clear heading, docstrings, and single-purpose cells |
| No Testing Activities | ✅ | No test files, pytest, assertions-as-tests; validation via demonstration only |
| Canonical ML & Viz Stack | ✅ | PyTorch for modeling; matplotlib + seaborn for loss curves; Hugging Face for training utilities |

---

## User Inputs Required

> **⚠️ INPUT REQUIRED**: The following items must be provided by the user before execution.

| Input | Description | Notebook Variable | Example |
|-------|-------------|-------------------|---------|
| **Recipe Dataset** | Plain text file, one recipe per line with [BOS]/[EOS] markers | `RECIPE_FILE_PATH` | `"recipes.txt"` |

```python
# ============================================================================
# SECTION 0: USER INPUTS (MODIFY THESE BEFORE RUNNING)
# ============================================================================
# 📁 Path to your recipe dataset file (upload to Colab runtime first)
RECIPE_FILE_PATH = "Dataset/structured_recipes_pretrain.txt"  # INPUT REQUIRED: Structured format with field markers
```

---

## Hyperparameters Configuration

> All hyperparameters are centralized in a single configuration cell for easy experimentation.

### Section 0.1: Tokenizer Hyperparameters

```python
# ============================================================================
# SECTION 0.1: TOKENIZER HYPERPARAMETERS
# ============================================================================
TOKENIZER_CONFIG = {
    "vocab_size": 12_000,           # Target vocabulary size for BPE (optimized for recipe corpus)
    "min_frequency": 2,             # Minimum token frequency to include
    "special_tokens": [
        "[PAD]",                    # Padding token (ID: 0)
        "[UNK]",                    # Unknown token (ID: 1)
        "[BOS]",                    # Beginning of sequence (ID: 2)
        "[EOS]",                    # End of sequence (ID: 3)
    ],
}
```

### Section 0.2: Model Architecture Hyperparameters

```python
# ============================================================================
# SECTION 0.2: MODEL ARCHITECTURE HYPERPARAMETERS (GPT-2 Mini)
# ============================================================================
MODEL_CONFIG = {
    "vocab_size": 12_000,           # CRITICAL: Must match TOKENIZER_CONFIG vocab_size exactly
    "n_positions": 3000,            # Maximum sequence length (context window)
    "n_embd": 512,                  # Embedding dimension
    "n_layer": 6,                   # Number of transformer layers
    "n_head": 8,                    # Number of attention heads (must divide n_embd)
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,             # Residual dropout
    "embd_pdrop": 0.1,              # Embedding dropout
    "attn_pdrop": 0.1,              # Attention dropout
}
# Approximate parameter count: ~50M
```

### Section 0.3: Training Hyperparameters

```python
# ============================================================================
# SECTION 0.3: TRAINING HYPERPARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    "num_train_epochs": 7,                     # Total training epochs (optimized for 8.6K samples)
    "per_device_train_batch_size": 4,          # Batch size per GPU (A100 40GB allows larger batches)
    "gradient_accumulation_steps": 4,          # Effective batch size = 4 * 4 = 16
    "learning_rate": 5e-5,                     # Peak learning rate
    "weight_decay": 0.01,                      # L2 regularization
    "warmup_steps": 500,                       # Linear warmup steps
    "fp16": True,                              # Mixed precision training
    "logging_dir": "./logs",                   # TensorBoard logs directory
    "logging_steps": 100,                      # Log every N steps
    "save_strategy": "epoch",                  # Save checkpoint every epoch
    "save_total_limit": 7,                     # Keep all 7 epoch checkpoints
    "output_dir": "./gpt2-recipe-checkpoints", # Checkpoint directory
    "report_to": "none",                       # Disable wandb/tensorboard
    "seed": 42,                                # Random seed for reproducibility
}
```

### Section 0.4: Inference Hyperparameters

```python
# ============================================================================
# SECTION 0.4: INFERENCE HYPERPARAMETERS
# ============================================================================
INFERENCE_CONFIG = {
    "max_new_tokens": 200,          # Maximum tokens to generate
    "temperature": 0.8,             # Sampling temperature (higher = more random)
    "top_k": 50,                    # Top-k sampling
    "top_p": 0.92,                  # Nucleus sampling threshold
    "do_sample": True,              # Enable sampling (vs greedy)
    "repetition_penalty": 1.1,      # Penalize repeated tokens
    "no_repeat_ngram_size": 3,      # Prevent repeating 3-grams
}
```

### Section 0.5: Fine-tuning Hyperparameters (Phase 2)

```python
# ============================================================================
# SECTION 0.5: FINE-TUNING HYPERPARAMETERS (PHASE 2)
# ============================================================================
FINETUNE_CONFIG = {
    "num_train_epochs": 3,                      # Fewer epochs for fine-tuning
    "per_device_train_batch_size": 2,           # Smaller batch for instruction data
    "gradient_accumulation_steps": 8,           # Effective batch size = 2 * 8 = 16
    "learning_rate": 1e-5,                      # Lower LR for fine-tuning (10x lower)
    "weight_decay": 0.01,                       # L2 regularization
    "warmup_ratio": 0.1,                        # 10% warmup (ratio-based)
    "fp16": True,                               # Mixed precision training
    "logging_dir": "./logs_finetune",           # Separate logs for fine-tuning
    "logging_steps": 50,                        # More frequent logging
    "save_strategy": "epoch",                   # Save checkpoint every epoch
    "save_total_limit": 3,                      # Keep last 3 checkpoints
    "output_dir": "./gpt2-recipe-instruct",     # Fine-tuned model checkpoints
    "report_to": "none",                        # Disable wandb/tensorboard
    "dataloader_num_workers": 2,                # Data loading workers
    "seed": 42,                                 # Random seed
    "max_seq_length": 1024,                     # Shorter sequences for instructions
}
```

### Section 0.6: Chatbot Configuration (Phase 3)

```python
# ============================================================================
# SECTION 0.6: CHATBOT CONFIGURATION (PHASE 3)
# ============================================================================
CHATBOT_CONFIG = {
    "model_path": "./model_finetuned",             # Path to fine-tuned model
    "default_temperature": 0.7,                 # Default sampling temperature
    "default_max_length": 300,                  # Default max generation tokens
    "temperature_range": (0.1, 1.0),            # UI slider range
    "max_length_range": (100, 1000),            # UI slider range
    "streamlit_port": 8501,                     # Default Streamlit port
    "page_title": "🍳 Recipe Chef AI",           # App title
    "page_icon": "🍳",                           # Browser tab icon
}
```
```

---

## Notebook Structure

> The notebook is organized into clearly delineated sections with headers, docstrings, and single-purpose cells.

```text
GPT2_Recipe_Pretraining.ipynb
│
├── SECTION 0: USER INPUTS & HYPERPARAMETERS
│   ├── 0.0 User Inputs (RECIPE_FILE_PATH)
│   ├── 0.1 Tokenizer Hyperparameters
│   ├── 0.2 Model Architecture Hyperparameters
│   ├── 0.3 Training Hyperparameters
│   └── 0.4 Inference Hyperparameters
│
├── SECTION 1: ENVIRONMENT SETUP
│   ├── 1.1 Install Dependencies (!pip install)
│   ├── 1.2 Import Libraries
│   └── 1.3 GPU Availability Check
│
├── SECTION 2: DATA LOADING
│   ├── 2.1 Load Recipe Text File
│   ├── 2.2 Data Exploration & Statistics
│   ├── 2.3 Visualize Recipe Length Distribution
│   └── 2.4 Validate Structured Format
│
├── SECTION 3: TOKENIZER TRAINING
│   ├── 3.1 Train Byte Pair Encoding (BPE) Tokenizer
│   ├── 3.2 Save Tokenizer to Disk
│   ├── 3.3 Wrap in GPT2TokenizerFast
│   └── 3.4 Tokenizer Validation (encode/decode demo)
│
├── SECTION 4: DATASET PREPARATION
│   ├── 4.1 RecipeDataset Class Definition
│   ├── 4.2 Tokenize and Prepare Dataset
│   └── 4.3 Create DataCollator
│
├── SECTION 5: MODEL INITIALIZATION
│   ├── 5.1 Configure GPT2Config
│   ├── 5.2 Initialize GPT2LMHeadModel (Random Weights)
│   └── 5.3 Model Summary & Parameter Count
│
├── SECTION 6: TRAINING
│   ├── 6.1 Configure TrainingArguments
│   ├── 6.2 Initialize Trainer
│   ├── 6.3 Execute Training Loop
│   └── 6.4 Visualize Training Loss Curve
│
├── SECTION 7: INFERENCE
│   ├── 7.1 Load Trained Checkpoint
│   ├── 7.2 Generate Recipe from Prompt
│   └── 7.3 Interactive Generation Examples
│
├── SECTION 8: CLEANUP & EXPORT
│   ├── 8.1 Save Final Model
│   └── 8.2 Download Artifacts (optional)
│
├── ═══════════════════════════════════════════════════════════════════════════
│   PHASE 2: INSTRUCTION FINE-TUNING (ALIGNMENT)
│   ═══════════════════════════════════════════════════════════════════════════
│
├── SECTION 9: PHASE 2 CONFIGURATION
│   ├── 9.1 Phase 2 User Inputs (INSTRUCTION_FILE_PATH)
│   └── 9.2 Fine-tuning Hyperparameters (FINETUNE_CONFIG)
│
├── SECTION 10: LOAD PHASE 1 ARTIFACTS
│   └── 10.1 Load Pre-trained Model & Tokenizer
│
├── SECTION 11: INSTRUCTION DATASET PREPARATION
│   ├── 11.1 Dataset Validation Function
│   ├── 11.2 Load and Validate Instruction Dataset
│   ├── 11.3 Instruction Formatting Function
│   ├── 11.4 InstructionDataset Class Definition
│   └── 11.5 Create Instruction Dataset
│
├── SECTION 12: INSTRUCTION FINE-TUNING
│   ├── 12.1 Configure Fine-tuning TrainingArguments
│   ├── 12.2 Initialize Fine-tuning Trainer
│   ├── 12.3 Execute Fine-tuning Loop
│   ├── 12.4 Visualize Fine-tuning Loss Curve
│   └── 12.5 Save Fine-tuned Model
│
├── SECTION 13: INTERACTIVE INFERENCE
│   ├── 13.1 Instruction-following Generation Function
│   ├── 13.2 Example Generations
│   └── 13.3 Interactive Chat Loop
│
└── SECTION 14: PHASE 2 EXPORT & SUMMARY
    ├── 14.1 Download Fine-tuned Model (Colab)
    └── 14.2 Complete Pipeline Summary
```

---

## Project Structure

### Documentation (this feature)

```text
specs/001-gpt2-pretrain-recipes/
├── spec.md              # Feature specification
├── plan.md              # This implementation plan
├── checklists/
│   ├── requirements.md  # Specification quality checklist
│   ├── libraries.md     # Python libraries checklist
│   └── hyperparameters.md # Hyperparameters checklist
└── tasks.md             # Implementation tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
notebooks/
└── GPT2_Recipe_Pretraining.ipynb    # Main Colab notebook (Phases 1-2)

app.py                               # Streamlit chatbot application (Phase 3)
colab_deploy.py                      # Colab deployment commands (Phase 3)
requirements.txt                     # Python dependencies for deployment

outputs/                              # Generated at runtime (gitignored)
├── tokenizer/                        # Trained BPE tokenizer files
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer_config.json
├── gpt2-recipe-checkpoints/          # Pre-trained model checkpoints
│   ├── checkpoint-epoch-1/
│   └── ...
└── gpt2-recipe-instruct/             # Fine-tuned model (Phase 2)
    └── pytorch_model.bin
```

**Structure Decision**: Jupyter Notebook for training (Phases 1-2); standalone Python files for deployment (Phase 3). Outputs are generated at runtime and stored in `outputs/` directory.

---

## Implementation Phases

### Phase 1: Environment & Data (Sections 1-2)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 1.1 | Install transformers, tokenizers via pip | ✅ Adds only spec-required deps |
| 1.2 | Import torch, transformers, matplotlib, seaborn | ✅ Canonical stack |
| 1.3 | Check GPU availability with torch.cuda | ✅ Clean, single-purpose |
| 2.1 | Load recipe text file from user-provided path | ✅ Clear input handling |
| 2.2 | Compute dataset statistics (count, lengths) | ✅ Informative exploration |
| 2.3 | Plot recipe length histogram with seaborn | ✅ Mandated viz stack |
| 2.4 | Validate structured format markers present | ✅ Data quality gate |

### Phase 2: Tokenizer (Section 3)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 3.1 | Train Byte Pair Encoding (BPE) tokenizer on corpus | ✅ From scratch, no pretrained |
| 3.2 | Save vocab.json, merges.txt to disk | ✅ Checkpoint artifacts |
| 3.3 | Wrap in GPT2TokenizerFast | ✅ HF compatibility |
| 3.4 | Demonstrate encode/decode with special tokens | ✅ Validation via demo, not test |

### Phase 3: Dataset & Model (Sections 4-5)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 4.1 | Define RecipeDataset(torch.utils.data.Dataset) | ✅ PyTorch native |
| 4.2 | Tokenize structured recipes, truncate/pad to n_positions tokens | ✅ Spec FR-007 |
| 4.3 | Create DataCollatorForLanguageModeling | ✅ HF utility |
| 5.1 | Configure GPT2Config with hyperparameters | ✅ Centralized config |
| 5.2 | Initialize GPT2LMHeadModel (random weights) | ✅ No pretrained |
| 5.3 | Print model summary and parameter count | ✅ Informative |

### Phase 4: Training (Section 6)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 6.1 | Configure TrainingArguments (FP16, grad accum) | ✅ Memory optimization |
| 6.2 | Initialize Trainer with model, dataset, args | ✅ HF Trainer |
| 6.3 | Execute trainer.train() for 7 epochs | ✅ Spec SC-002 |
| 6.4 | Plot loss curve with matplotlib/seaborn | ✅ Mandated viz |

### Phase 5: Inference (Section 7)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 7.1 | Load checkpoint from disk | ✅ Spec FR-012 |
| 7.2 | Generate recipe from "Ingredients: Chicken" | ✅ Spec SC-004 |
| 7.3 | Provide interactive examples | ✅ Demonstration-based |

### Phase 6: Export (Section 8)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 8.1 | Save final model to disk | ✅ Checkpoint artifacts |
| 8.2 | Zip and download artifacts | ✅ Colab-compatible |

### Phase 7: Instruction Fine-tuning Setup (Sections 9-10)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 9.1 | Configure instruction file path | ✅ Clear user input |
| 9.2 | Define FINETUNE_CONFIG hyperparameters | ✅ Centralized config |
| 10.1 | Load pre-trained model and tokenizer | ✅ Spec FR-013 |

### Phase 8: Instruction Dataset (Section 11)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 11.1 | Define validation function for Alpaca JSONL | ✅ Spec FR-014 |
| 11.2 | Load and validate instruction dataset | ✅ Error reporting |
| 11.3 | Define format_instruction() with template | ✅ Spec FR-015 |
| 11.4 | Define InstructionDataset class | ✅ Spec FR-016, PyTorch native |
| 11.5 | Create instruction dataset instance | ✅ Dataset preparation |

### Phase 9: Fine-tuning (Section 12)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 12.1 | Configure TrainingArguments (lower LR) | ✅ Spec FR-017 |
| 12.2 | Initialize Trainer for fine-tuning | ✅ HF Trainer |
| 12.3 | Execute fine-tuning for 3 epochs | ✅ Spec SC-006 |
| 12.4 | Plot fine-tuning loss curve | ✅ Mandated viz |
| 12.5 | Save fine-tuned model | ✅ Checkpoint artifacts |

### Phase 10: Interactive Inference (Section 13)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 13.1 | Define generate_from_instruction() | ✅ Spec FR-018 |
| 13.2 | Demonstrate with example instructions | ✅ Spec SC-008 |
| 13.3 | Provide interactive chat loop | ✅ Demonstration-based |

### Phase 11: Final Export (Section 14)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 14.1 | Download fine-tuned model artifacts | ✅ Colab-compatible |
| 14.2 | Display complete pipeline summary | ✅ Informative |

### Phase 8: Chatbot Deployment (app.py, colab_deploy.py)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 12.1 | Create Streamlit app with professional UI | ✅ Standalone Python file |
| 12.2 | Implement model loading with @st.cache_resource | ✅ Memory efficient |
| 12.3 | Add sidebar generation controls (temperature, length) | ✅ User configurable |
| 12.4 | Implement chat interface with st.chat_message | ✅ Clean UX |
| 12.5 | Create Colab deployment script with ngrok tunnel | ✅ Accessible from web |
| 12.6 | Create requirements.txt for dependencies | ✅ Reproducible environment |

---

## Complexity Tracking

> No constitution violations. No complexity justifications required.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | — | — |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tokenizer training time | <5 minutes | Cell execution time |
| Phase 1 training completion | 7 epochs, no OOM | Trainer logs |
| Phase 1 loss trend | final_loss < initial_loss | Loss curve visualization |
| Pre-training generation quality | Structured output with labeled sections | Manual inspection for `**Title:**`, `**Ingredients:**`, `**Instructions:**` |
| Phase 1 runtime | <2 hours | Notebook execution time |
| Phase 2 fine-tuning completion | 3 epochs, no OOM | Trainer logs |
| Phase 2 loss trend | final_loss < initial_loss | Loss curve visualization |
| Instruction-following quality | Structured recipe with bullets/numbered steps | Manual inspection |
| Total pipeline runtime | <3 hours | End-to-end execution time |
| Chatbot model load time | <30 seconds | Streamlit first request |
| Ngrok tunnel setup | <10 seconds | Deployment script execution |
| Chat response latency | <10 seconds for 300 tokens | Manual testing on GPU |
| Session persistence | 5+ messages retained | Chat history inspection |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OOM on A100 GPU | Low | A100 40GB provides ample headroom; reduce batch size if needed |
| Colab disconnection | High | Save checkpoints every epoch; resume from latest |
| Poor generation quality | Medium | Ensure 7 epochs complete; tune temperature/top_p |
| Tokenizer OOV issues | Low | 12K vocab with min_frequency=2 covers domain |

---

## Next Steps

1. **Create tasks.md**: Run `/speckit.tasks` to generate implementation task list
2. **Create notebook**: Implement sections 0-14 following this plan
3. **Execute Phase 1 on Colab**: Upload `Dataset/structured_recipes_pretrain.txt` and run pre-training (Sections 0-8)
4. **Execute Phase 2 on Colab**: Upload `Dataset/structured_recipes_finetune.jsonl` and run instruction fine-tuning (Sections 9-14)
5. **Deploy chatbot**: Run `colab_deploy.py` to launch Streamlit app with ngrok tunnel
6. **Validate success criteria**: Confirm all SC-001 through SC-013 metrics, especially **structured output format**
