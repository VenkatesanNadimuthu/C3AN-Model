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

### User Story 5 - Deploy Interactive Chatbot Application (Priority: P5)

A user deploys the fine-tuned recipe model as an interactive web chatbot using Streamlit, accessible via a public URL from Google Colab using ngrok tunneling.

**Why this priority**: After training and fine-tuning, deployment makes the model accessible to end users. A polished chatbot interface demonstrates practical value and enables sharing.

**Independent Demonstration**: Run Colab deployment cells; access the ngrok public URL; interact with the chatbot by typing recipe requests; verify responses appear in chat bubble format.

**Acceptance Scenarios**:

1. **Given** a fine-tuned model saved to disk, **When** the user runs the Streamlit app, **Then** the model loads once via caching and responds to user messages in under 5 seconds.
2. **Given** the Streamlit app running on Colab, **When** the user configures ngrok with an auth token, **Then** a public HTTPS URL is generated that external users can access.
3. **Given** the chatbot interface, **When** the user adjusts temperature and max length sliders, **Then** subsequent generations reflect the updated parameters.
4. **Given** a conversation in progress, **When** the user sends multiple messages, **Then** all messages remain visible in the chat history (session state persists).

---

### Edge Cases

- What happens when a recipe row exceeds 3000 characters? Rows longer than the max sequence length are truncated during tokenization; padding is applied to shorter rows.
- How does the system handle GPU unavailability? The environment-setup cell checks for GPU; if none is detected, a warning is printed and training proceeds on CPU (slower but functional).
- What if the tokenizer encounters a character not seen during training? The [UNK] special token is used; the tokenizer vocabulary includes a fallback.
- How does the system handle space-padding in input data? The tokenizer treats spaces as regular tokens; attention masks ensure padded positions do not influence loss.
- What if the instruction JSONL file has malformed entries? The validation function reports line numbers and error types; only valid samples are used for training.
- What if the pre-trained model checkpoint is missing? The system raises FileNotFoundError with instructions to complete Phase 1 first.
- What if ngrok auth token is invalid or missing? The tunnel creation fails with a clear error message; user is directed to https://dashboard.ngrok.com to obtain a valid token.
- What if the Streamlit app crashes due to memory? The model is loaded with `@st.cache_resource` to prevent repeated loading; GPU memory is released between sessions.
- What if multiple users access the chatbot simultaneously? Each user session maintains independent chat history via `st.session_state`; model inference is serialized.

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

### Functional Requirements (Phase 3: Chatbot Deployment)

- **FR-019**: System MUST provide a Streamlit application (`app.py`) with a professional chat interface using `st.chat_message` for user/assistant message bubbles.
- **FR-020**: System MUST use `@st.cache_resource` to load the model and tokenizer exactly once, preventing Colab memory issues on repeated interactions.
- **FR-021**: System MUST provide sidebar controls for generation parameters (temperature slider 0.1-1.0, max length slider 100-1000) that affect subsequent generations.
- **FR-022**: System MUST maintain chat history in `st.session_state` so conversations persist across user interactions within a session.
- **FR-023**: System MUST provide Colab deployment commands (`colab_deploy.py`) that install dependencies, configure ngrok tunneling, and launch Streamlit with a public URL.
- **FR-024**: System MUST automatically detect and use GPU (CUDA) if available, falling back to CPU for inference.

### Key Entities

- **Recipe**: A single training example on one line comprising Recipe Name, Ingredients, Instructions concatenated with [BOS] prefix and [EOS] suffix; max 3000 characters; space-padded.
- **Tokenizer**: A Byte Pair Encoding (BPE) tokenizer trained on the recipe corpus; vocabulary size of 30,000 tokens; includes special tokens [BOS], [EOS], [UNK], [PAD].
- **Model**: A GPT-2 Mini architecture (6 layers, 512 embedding dimension, 8 attention heads, ~50M parameters) initialized with random weights.
- **Checkpoint**: Serialized model weights and tokenizer files stored on disk after each epoch (10 total), loadable for inference.
- **Data Source**: Recipe text file uploaded directly to Colab runtime.
- **Instruction Sample**: A single fine-tuning example in Alpaca-style JSONL format with `instruction` field (user request) and `response` field (recipe content); one JSON object per line.
- **Fine-tuned Model**: The instruction-aligned model saved after Phase 2, capable of following user instructions to generate recipes.
- **Streamlit App**: A Python web application (`app.py`) providing chat interface, sidebar configuration, and session-based conversation history.
- **Ngrok Tunnel**: A secure HTTPS tunnel exposing the local Streamlit server (port 8501) to a public URL for external access from Colab.
- **Chat Session**: A user interaction session maintaining message history, generation parameters, and model state via Streamlit session_state.

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

### Measurable Outcomes (Phase 3: Chatbot Deployment)

- **SC-010**: Streamlit app loads the fine-tuned model in under 30 seconds on first request; subsequent requests use cached model with no reload.
- **SC-011**: Ngrok tunnel is established within 10 seconds of running deployment commands, providing a valid HTTPS public URL.
- **SC-012**: Chat response latency is under 10 seconds for a typical recipe generation request (300 tokens) on GPU.
- **SC-013**: Chat history persists correctly—sending 5 consecutive messages results in all 5 user messages and 5 assistant responses visible in the interface.
