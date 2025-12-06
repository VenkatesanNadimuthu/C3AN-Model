# Implementation Plan: GPT-2 Pre-training on Recipe Dataset

**Branch**: `001-gpt2-pretrain-recipes` | **Date**: 2025-12-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-gpt2-pretrain-recipes/spec.md`
**Author**: AI Research Engineer

---

## Summary

Build and pre-train a GPT-2 Mini language model (~50M parameters) from scratch on a domain-specific recipe dataset containing 8,500+ entries. The implementation trains a custom Byte Pair Encoding tokenizer, initializes the transformer architecture with random weights, and executes 10 epochs of causal language modeling using Hugging Face Trainer with FP16 mixed precision. The entire pipeline is optimized for Google Colab's A100 GPU (≈40 GB VRAM) and produces a checkpoint capable of generating coherent recipe text from prompts.

---

## Technical Context

**Language/Version**: Python 3.10+ (Colab default)  
**Primary Dependencies**: PyTorch 2.x, transformers 4.x, tokenizers 0.x  
**Storage**: Local filesystem (Colab runtime) for checkpoints and tokenizer artifacts  
**Testing**: Not permitted by constitution (no automated or manual test tasks)  
**Target Platform**: Google Colab with NVIDIA A100 GPU  
**Project Type**: Single Jupyter Notebook (Colab-native)  
**Performance Goals**: Complete 10 epochs on 8,500 recipes in <2 hours; generate coherent 50+ token continuations  
**Constraints**: ≈40 GB GPU memory; 3000-token max sequence length; FP16 required  
**Scale/Scope**: ~50M parameter model; 30K vocabulary; 8,500 training samples

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
RECIPE_FILE_PATH = "recipes.txt"  # INPUT REQUIRED: Set your file path here
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
    "vocab_size": 30_000,           # Target vocabulary size for BPE
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
    "vocab_size": 30_000,           # Must match tokenizer vocab_size
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
    "num_train_epochs": 10,                    # Total training epochs
    "per_device_train_batch_size": 4,          # Batch size per GPU (A100 40GB allows larger batches)
    "gradient_accumulation_steps": 4,          # Effective batch size = 4 * 4 = 16
    "learning_rate": 5e-5,                     # Peak learning rate
    "weight_decay": 0.01,                      # L2 regularization
    "warmup_steps": 500,                       # Linear warmup steps
    "fp16": True,                              # Mixed precision training
    "logging_steps": 100,                      # Log every N steps
    "save_strategy": "epoch",                  # Save checkpoint every epoch
    "save_total_limit": 10,                    # Keep all 10 epoch checkpoints
    "output_dir": "./gpt2-recipe-checkpoints", # Checkpoint directory
    "report_to": "none",                       # Disable wandb/tensorboard
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
}
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
│   └── 2.3 Visualize Recipe Length Distribution
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
└── SECTION 8: CLEANUP & EXPORT
    ├── 8.1 Save Final Model
    └── 8.2 Download Artifacts (optional)
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
└── GPT2_Recipe_Pretraining.ipynb    # Main Colab notebook

outputs/                              # Generated at runtime (gitignored)
├── tokenizer/                        # Trained BPE tokenizer files
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer_config.json
└── gpt2-recipe-checkpoints/          # Model checkpoints (epoch 1-10)
    ├── checkpoint-epoch-1/
    ├── checkpoint-epoch-2/
    └── ...
```

**Structure Decision**: Single Jupyter Notebook for Colab execution. All code is contained in one `.ipynb` file with clear section headers. Outputs (tokenizer, checkpoints) are generated at runtime and stored in `outputs/` directory.

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
| 4.2 | Tokenize, truncate/pad to 3000 tokens | ✅ Spec FR-007 |
| 4.3 | Create DataCollatorForLanguageModeling | ✅ HF utility |
| 5.1 | Configure GPT2Config with hyperparameters | ✅ Centralized config |
| 5.2 | Initialize GPT2LMHeadModel (random weights) | ✅ No pretrained |
| 5.3 | Print model summary and parameter count | ✅ Informative |

### Phase 4: Training (Section 6)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 6.1 | Configure TrainingArguments (FP16, grad accum) | ✅ Memory optimization |
| 6.2 | Initialize Trainer with model, dataset, args | ✅ HF Trainer |
| 6.3 | Execute trainer.train() for 10 epochs | ✅ Spec SC-002 |
| 6.4 | Plot loss curve with matplotlib/seaborn | ✅ Mandated viz |

### Phase 5: Inference (Section 7)

| Step | Description | Constitution Compliance |
|------|-------------|------------------------|
| 7.1 | Load checkpoint from disk | ✅ Spec FR-012 |
| 7.2 | Generate recipe from "Ingredients: Chicken" | ✅ Spec SC-004 |
| 7.3 | Provide interactive examples | ✅ Demonstration-based |

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
| Training completion | 10 epochs, no OOM | Trainer logs |
| Loss trend | final_loss < initial_loss | Loss curve visualization |
| Generation quality | 50+ coherent tokens | Manual inspection |
| Total runtime | <2 hours | Notebook execution time |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OOM on A100 GPU | Low | A100 40GB provides ample headroom; reduce batch size if needed |
| Colab disconnection | High | Save checkpoints every epoch; resume from latest |
| Poor generation quality | Medium | Ensure 10 epochs complete; tune temperature/top_p |
| Tokenizer OOV issues | Low | 30K vocab with min_frequency=2 covers domain |

---

## Next Steps

1. **Create tasks.md**: Run `/speckit.tasks` to generate implementation task list
2. **Create notebook**: Implement sections 0-8 following this plan
3. **Execute on Colab**: Upload recipe dataset and run end-to-end
4. **Validate success criteria**: Confirm all SC-001 through SC-005 metrics
