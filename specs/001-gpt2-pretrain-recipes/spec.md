# Feature Specification: GPT-2 Pre-training on Recipe Dataset

**Feature Branch**: `001-gpt2-pretrain-recipes`  
**Created**: 2025-12-06  
**Status**: Draft  
**Input**: User description: "Build and pre-train a GPT-2 Language Model from scratch using a custom recipe dataset, optimized for Google Colab GPU execution"

## User Scenarios & Demonstrations *(mandatory, no automated testing)*

### User Story 1 - Train Custom BPE Tokenizer on Recipe Corpus (Priority: P1)

A researcher loads a recipe dataset (8,500+ rows containing Recipe Name, Ingredients, Instructions with [BOS]/[EOS] markers) from a plain text file and trains a Byte Pair Encoding (BPE) tokenizer from scratch that learns the vocabulary specific to culinary text.

**Why this priority**: Without a tokenizer tailored to recipe terminology, the model cannot encode domain-specific vocabulary (ingredient names, cooking verbs, measurements) effectively. This is the foundational step for all downstream work.

**Independent Demonstration**: Run a single notebook cell that loads the text file, trains the BPE tokenizer, saves it to disk, and encodes/decodes a sample recipe text showing that special tokens [BOS], [EOS], [UNK], [PAD] are correctly handled.

**Acceptance Scenarios**:

1. **Given** a text file with 8,500+ recipe entries (Recipe Name, Ingredients, Instructions) where each entry is pre-marked with [BOS] and [EOS] tokens, **When** the user executes the tokenizer training cell, **Then** a new BPE tokenizer is saved to disk and can encode any recipe text into token IDs and decode back to the original string.
2. **Given** a trained tokenizer, **When** the user encodes text containing [BOS], [EOS], or unknown characters, **Then** the special tokens map to their designated IDs and unknown tokens map to [UNK].

---

### User Story 2 - Initialize and Pre-train GPT-2 from Random Weights (Priority: P2)

A researcher configures a GPT-2 architecture from scratch (no pre-trained weights) and pre-trains it on the tokenized recipe corpus using the Hugging Face Trainer with FP16 and gradient accumulation to fit within Colab GPU memory constraints.

**Why this priority**: Training the language model is the core objective. With the tokenizer in place, the model learns to predict recipe continuations, enabling downstream generation tasks.

**Independent Demonstration**: Execute the training loop for 10 epochs; observe decreasing loss logged by Trainer; save the trained checkpoint to disk.

**Acceptance Scenarios**:

1. **Given** a tokenized dataset and a randomly initialized GPT-2 model, **When** the user starts the Trainer for 10 epochs, **Then** training progresses using FP16 mixed precision, and loss decreases over epochs.
2. **Given** A100 GPU memory (≈40 GB), **When** training with sequences up to 3000 tokens for 10 epochs, **Then** gradient accumulation keeps memory usage within limits and training completes without OOM errors.

---

### User Story 3 - Generate New Recipe Text via Inference (Priority: P3)

A user loads the trained model and tokenizer, provides a prompt such as "Ingredients: Chicken", and generates a coherent recipe continuation.

**Why this priority**: Generation demonstrates that the model has learned useful representations. It is the user-facing payoff of the training effort.

**Independent Demonstration**: Run an inference cell that loads the checkpoint, encodes a prompt, calls `model.generate()`, and decodes the output to human-readable recipe text.

**Acceptance Scenarios**:

1. **Given** a trained GPT-2 checkpoint and tokenizer, **When** the user provides the prompt "Ingredients: Chicken", **Then** the model generates text that resembles recipe instructions continuing from that prompt.
2. **Given** a prompt shorter than 3000 tokens, **When** generation is invoked, **Then** output respects max length and includes an [EOS] token or stops gracefully.

---

### User Story 4 - Instruction Fine-tune for Recipe Generation (Priority: P4)

A user fine-tunes the pre-trained recipe model to follow natural language instructions using an Alpaca-style dataset, enabling conversational recipe generation.

**Why this priority**: After pre-training learns domain knowledge, instruction fine-tuning aligns the model to follow user requests, making it practically useful as a recipe assistant.

**Independent Demonstration**: Run Phase 2 cells; provide instruction prompts like "Give me a recipe for chocolate cake"; verify model generates relevant recipe responses in the expected format.

**Acceptance Scenarios**:

1. **Given** a pre-trained GPT-2 recipe model and an Alpaca-style JSONL dataset with `instruction` and `response` fields, **When** the user executes fine-tuning for 3 epochs with lower learning rate (1e-5), **Then** the model learns to generate responses following the `### Instruction: / ### Response:` template.
2. **Given** a fine-tuned model, **When** the user provides an instruction like "How do I make pasta carbonara?", **Then** the model generates a coherent recipe response that addresses the specific request.
3. **Given** an instruction dataset with validation errors, **When** the user loads the dataset, **Then** the system reports specific errors (missing fields, empty content, invalid JSON) and skips invalid entries.

---

### Edge Cases

- What happens when a recipe row exceeds 3000 characters? Rows longer than the max sequence length are truncated during tokenization; padding is applied to shorter rows.
- How does the system handle GPU unavailability? The environment-setup cell checks for GPU; if none is detected, a warning is printed and training proceeds on CPU (slower but functional).
- What if the tokenizer encounters a character not seen during training? The [UNK] special token is used; the tokenizer vocabulary includes a fallback.
- How does the system handle space-padding in input data? The tokenizer treats spaces as regular tokens; attention masks ensure padded positions do not influence loss.
- What if the instruction JSONL file has malformed entries? The validation function reports line numbers and error types; only valid samples are used for training.
- What if the pre-trained model checkpoint is missing? The system raises FileNotFoundError with instructions to complete Phase 1 first.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST install required packages (transformers, datasets, tokenizers) via pip commands executable in Google Colab.
- **FR-002**: System MUST verify GPU availability and report the device name; fall back to CPU if no GPU is present.
- **FR-003**: System MUST load recipe data from a plain text file where each line contains one complete recipe (Recipe Name, Ingredients, Instructions) with [BOS]/[EOS] markers.
- **FR-004**: System MUST train a new Byte Pair Encoding (BPE) tokenizer on the recipe corpus without using any pre-trained tokenizer with a target vocabulary size of 30,000 tokens.
- **FR-005**: System MUST register special tokens [BOS], [EOS], [UNK], [PAD] in the tokenizer and save the tokenizer to disk.
- **FR-006**: System MUST wrap the trained tokenizer in GPT2TokenizerFast for compatibility with Hugging Face models.
- **FR-007**: System MUST implement a custom PyTorch Dataset class that tokenizes recipes, truncates/pads to a maximum length of 3000 tokens, and returns input_ids, attention_mask, and labels.
- **FR-008**: System MUST initialize a GPT-2 Mini model from scratch using GPT2Config with 6 layers, 512 embedding dimension, and 8 attention heads (random weights, no pre-trained checkpoint).
- **FR-009**: System MUST train the model using Hugging Face Trainer with FP16 (mixed precision) enabled.
- **FR-010**: System MUST use gradient accumulation to allow effective batch sizes that fit within Colab GPU memory while handling 3000-token sequences.
- **FR-011**: System MUST save model checkpoints to disk after every epoch (10 checkpoints total).
- **FR-012**: System MUST provide an inference routine that loads the saved model and tokenizer, accepts a text prompt, and generates continuation text.

### Functional Requirements (Phase 2: Instruction Fine-tuning)

- **FR-013**: System MUST load a pre-trained model and tokenizer from Phase 1 output directory.
- **FR-014**: System MUST validate Alpaca-style JSONL instruction datasets, reporting errors for missing `instruction` or `response` fields, empty content, or invalid JSON.
- **FR-015**: System MUST format instruction-response pairs using the template: `### Instruction:\n{instruction}\n\n### Response:\n{response}[EOS]`.
- **FR-016**: System MUST implement an InstructionDataset class that formats and tokenizes instruction-response pairs on-the-fly.
- **FR-017**: System MUST fine-tune using a lower learning rate (1e-5) than pre-training to preserve domain knowledge while learning instruction-following.
- **FR-018**: System MUST provide an instruction-following inference function that formats user input as an instruction prompt and extracts only the generated response.

### Key Entities

- **Recipe**: A single training example on one line comprising Recipe Name, Ingredients, Instructions concatenated with [BOS] prefix and [EOS] suffix; max 3000 characters; space-padded.
- **Tokenizer**: A Byte Pair Encoding (BPE) tokenizer trained on the recipe corpus; vocabulary size of 30,000 tokens; includes special tokens [BOS], [EOS], [UNK], [PAD].
- **Model**: A GPT-2 Mini architecture (6 layers, 512 embedding dimension, 8 attention heads, ~50M parameters) initialized with random weights.
- **Checkpoint**: Serialized model weights and tokenizer files stored on disk after each epoch (10 total), loadable for inference.
- **Data Source**: Recipe text file uploaded directly to Colab runtime.
- **Instruction Sample**: A single fine-tuning example in Alpaca-style JSONL format with `instruction` field (user request) and `response` field (recipe content); one JSON object per line.
- **Fine-tuned Model**: The instruction-aligned model saved after Phase 2, capable of following user instructions to generate recipes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tokenizer training completes in under 5 minutes on Colab and produces a vocabulary that encodes any recipe in the dataset without errors.
- **SC-002**: Model training runs for 10 full epochs on 8,500 recipes without out-of-memory errors on a Colab A100 GPU.
- **SC-003**: Training loss decreases over the course of training (final loss < initial loss).
- **SC-004**: Inference generates at least 50 new tokens of coherent recipe-style text given the prompt "Ingredients: Chicken". Coherent is defined as: output contains at least one ingredient name AND at least one cooking verb (e.g., "cook", "bake", "mix", "stir", "add").
- **SC-005**: End-to-end notebook execution (setup → tokenizer → dataset → training → inference) completes in under 2 hours on Colab A100 GPU with the 8,500-recipe dataset.

### Measurable Outcomes (Phase 2: Instruction Fine-tuning)

- **SC-006**: Instruction fine-tuning runs for 3 full epochs without out-of-memory errors on Colab A100 GPU.
- **SC-007**: Fine-tuning loss decreases over the course of training (final loss < initial loss).
- **SC-008**: Given the instruction "Give me a recipe for chocolate cake", the model generates a response containing at least one ingredient AND at least one cooking step.
- **SC-009**: The instruction-following inference function correctly extracts only the response portion (after `### Response:`), excluding the instruction prompt from output.
