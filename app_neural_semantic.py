"""
🍳 Gourmet GPT: Neural Semantic Recipe Search
Uses GPT-2 embeddings for true semantic understanding and similarity matching.

Author: AI Research Engineer
Date: 2025-12-08
"""

import streamlit as st
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gourmet GPT: Neural Semantic Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    .main-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    /* Pipeline step cards */
    .pipeline-step {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        margin: 0.5rem 0; border-left: 4px solid #11998e;
    }
    .pipeline-step.embedding { border-left-color: #6f42c1; }
    .pipeline-step.similarity { border-left-color: #fd7e14; }
    .pipeline-step.ranking { border-left-color: #17a2b8; }
    
    .step-title { font-weight: 600; color: #333; margin-bottom: 0.5rem; }
    .step-content { color: #555; }
    
    /* Similarity score badge */
    .similarity-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.85rem; display: inline-block; margin-right: 0.5rem;
    }
    
    /* Recipe card */
    .recipe-card {
        background: white; border-radius: 10px; padding: 1.5rem;
        margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e8e8e8;
    }
    .recipe-card h3 { color: #333; margin-bottom: 0.5rem; }
    
    .recipe-section {
        margin: 1rem 0; padding: 0.75rem;
        background: #f8f9fa; border-radius: 8px;
        border-left: 3px solid #11998e;
    }
    .recipe-section h4 { color: #11998e; margin: 0 0 0.5rem 0; }
    .recipe-section ul, .recipe-section ol { margin: 0.5rem 0 0 1.5rem; padding: 0; }
    .recipe-section li { margin: 0.3rem 0; }
    
    /* Vector visualization */
    .vector-viz {
        background: #1a1a2e; color: #00ff88; padding: 0.5rem;
        border-radius: 5px; font-family: monospace; font-size: 0.75rem;
        overflow-x: auto; white-space: nowrap;
    }
    
    .footer { text-align: center; padding: 1rem; color: #6c757d; font-size: 0.85rem; margin-top: 2rem; border-top: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Recipe:
    """A recipe with its embedding."""
    title: str
    ingredients: list
    instructions: list
    serving_suggestion: str
    tip: str
    text: str  # Combined searchable text
    embedding: Optional[np.ndarray] = None


@dataclass
class SemanticMatch:
    """A recipe match with similarity score."""
    recipe: Recipe
    similarity: float
    rank: int


# ============================================================================
# GPT-2 EMBEDDING EXTRACTOR
# ============================================================================

class GPT2EmbeddingExtractor:
    """
    Extract semantic embeddings from GPT-2's hidden states.
    
    GPT-2 is a decoder-only transformer, but we can use its hidden states
    as semantic representations by:
    1. Passing text through the model
    2. Extracting the last hidden state
    3. Mean pooling across tokens to get a fixed-size vector
    """
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = model.config.n_embd  # 768 for GPT-2 base
        
    @torch.no_grad()
    def get_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Extract embedding for a single text.
        
        Uses mean pooling of the last hidden state across all tokens.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        # Forward pass with hidden states
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        
        # Get last hidden state: (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = last_hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1)
        mean_pooled = sum_hidden / count
        
        # Normalize for cosine similarity
        embedding = F.normalize(mean_pooled, p=2, dim=-1)
        
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def get_embeddings_batch(self, texts: list[str], batch_size: int = 16, max_length: int = 256) -> np.ndarray:
        """
        Extract embeddings for multiple texts efficiently.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            
            # Mean pooling
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            count = attention_mask.sum(dim=1)
            mean_pooled = sum_hidden / count
            
            # Normalize
            embeddings = F.normalize(mean_pooled, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


# ============================================================================
# MODEL & EMBEDDINGS LOADING
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer(model_path: str = "./model_finetuned"):
    """Load GPT-2 model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


@st.cache_data
def load_recipes_from_jsonl(filepath: str = "./Dataset/structured_recipes_finetune.jsonl") -> list[dict]:
    """Load and parse recipes from JSONL file."""
    recipes = []
    seen_titles = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    response = data.get('response', '')
                    recipe = parse_recipe_response(response)
                    
                    if recipe and recipe['title'] not in seen_titles:
                        seen_titles.add(recipe['title'])
                        recipes.append(recipe)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        st.error(f"Recipe file not found: {filepath}")
    
    return recipes


def parse_recipe_response(response: str) -> Optional[dict]:
    """Parse structured recipe response."""
    import re
    
    recipe = {
        'title': '',
        'ingredients': [],
        'instructions': [],
        'serving_suggestion': '',
        'tip': '',
    }
    
    # Extract title
    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n|$)', response)
    if title_match:
        recipe['title'] = title_match.group(1).strip()
    else:
        return None
    
    # Extract ingredients
    ingredients_match = re.search(
        r'\*\*Ingredients:\*\*\s*\n?([\s\S]*?)(?=\*\*Instructions:|\*\*Serving|\*\*Tip:|$)',
        response
    )
    if ingredients_match:
        items = re.findall(r'-\s*(.+?)(?=\s*-\s|\s*\*\*|$)', ingredients_match.group(1))
        recipe['ingredients'] = [item.strip() for item in items if item.strip()]
    
    # Extract instructions
    instructions_match = re.search(
        r'\*\*Instructions:\*\*\s*\n?([\s\S]*?)(?=\*\*Serving|\*\*Tip:|$)',
        response
    )
    if instructions_match:
        steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', instructions_match.group(1), re.DOTALL)
        recipe['instructions'] = [step.strip() for step in steps if step.strip()]
    
    # Extract serving suggestion
    serving_match = re.search(r'\*\*Serving Suggestion:\*\*\s*(.+?)(?:\n|$)', response)
    if serving_match:
        recipe['serving_suggestion'] = serving_match.group(1).strip()
    
    # Extract tip
    tip_match = re.search(r'\*\*Tip:\*\*\s*(.+?)(?:\n|$)', response)
    if tip_match:
        recipe['tip'] = tip_match.group(1).strip()
    
    # Create combined text for embedding
    recipe['text'] = f"{recipe['title']}. Ingredients: {', '.join(recipe['ingredients'][:15])}. {' '.join(recipe['instructions'][:3])}"
    
    return recipe


def get_model_artifacts_path(model_path: str) -> tuple[Path, Path]:
    """Get paths for pre-computed embeddings in model artifacts."""
    model_dir = Path(model_path)
    embeddings_path = model_dir / "recipe_embeddings.npz"
    metadata_path = model_dir / "recipes_metadata.json"
    return embeddings_path, metadata_path


@st.cache_data
def load_precomputed_embeddings(model_path: str) -> tuple[np.ndarray, list[dict]] | None:
    """
    Load pre-computed embeddings from model artifacts.
    
    Returns:
        Tuple of (embeddings, recipes) if found, None otherwise.
    """
    embeddings_path, metadata_path = get_model_artifacts_path(model_path)
    
    if not embeddings_path.exists() or not metadata_path.exists():
        return None
    
    try:
        # Load embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data['embeddings']
        
        # Load recipe metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        return embeddings, recipes
    except Exception as e:
        st.warning(f"Error loading pre-computed embeddings: {e}")
        return None


def compute_embeddings_on_demand(
    recipes: list[dict],
    extractor: GPT2EmbeddingExtractor,
    model_path: str,
    progress_callback=None
) -> np.ndarray:
    """
    Compute embeddings if not pre-computed.
    Also saves them to model artifacts for next time.
    """
    texts = [r['text'] for r in recipes]
    
    # Process in batches with progress
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = extractor.get_embeddings_batch(batch_texts, batch_size=len(batch_texts))
        all_embeddings.append(batch_embeddings)
        
        if progress_callback:
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_callback(progress, f"Processed {min(i + batch_size, len(texts))}/{len(texts)} recipes")
    
    embeddings = np.vstack(all_embeddings)
    
    # Save as model artifacts for future use
    embeddings_path, metadata_path = get_model_artifacts_path(model_path)
    
    try:
        titles = [r['title'] for r in recipes]
        texts_arr = [r['text'] for r in recipes]
        
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            titles=np.array(titles, dtype=object),
            texts=np.array(texts_arr, dtype=object)
        )
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False)
        
        st.success(f"✅ Embeddings saved to model artifacts for future use!")
    except Exception as e:
        st.warning(f"Could not save embeddings to model artifacts: {e}")
    
    return embeddings


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

def semantic_search(
    query: str,
    query_embedding: np.ndarray,
    recipe_embeddings: np.ndarray,
    recipes: list[dict],
    top_k: int = 10
) -> list[SemanticMatch]:
    """
    Find most similar recipes using cosine similarity.
    
    Since embeddings are already normalized, dot product = cosine similarity.
    """
    # Compute similarities
    similarities = np.dot(recipe_embeddings, query_embedding)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        recipe_dict = recipes[idx]
        recipe = Recipe(
            title=recipe_dict['title'],
            ingredients=recipe_dict['ingredients'],
            instructions=recipe_dict['instructions'],
            serving_suggestion=recipe_dict.get('serving_suggestion', ''),
            tip=recipe_dict.get('tip', ''),
            text=recipe_dict['text']
        )
        results.append(SemanticMatch(
            recipe=recipe,
            similarity=float(similarities[idx]),
            rank=rank
        ))
    
    return results


# ============================================================================
# UI RENDERING
# ============================================================================

def render_pipeline_step(step_type: str, title: str, content: str):
    """Render a pipeline step card."""
    st.markdown(f"""
    <div class="pipeline-step {step_type}">
        <div class="step-title">{title}</div>
        <div class="step-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_vector_preview(embedding: np.ndarray, num_dims: int = 10):
    """Render a preview of the embedding vector."""
    preview = ", ".join([f"{v:.3f}" for v in embedding[:num_dims]])
    st.markdown(f"""
    <div class="vector-viz">
        [{preview}, ... ] (768 dimensions)
    </div>
    """, unsafe_allow_html=True)


def render_recipe_card(match: SemanticMatch):
    """Render a recipe card with similarity score."""
    recipe = match.recipe
    similarity_pct = match.similarity * 100
    
    # Color gradient based on similarity
    if similarity_pct >= 70:
        color = "#28a745"
    elif similarity_pct >= 50:
        color = "#ffc107"
    else:
        color = "#6c757d"
    
    st.markdown(f"""
    <div class="recipe-card">
        <span class="similarity-badge" style="background: {color};">
            #{match.rank} · {similarity_pct:.1f}% match
        </span>
        <h3>🍽️ {recipe.title}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 View Full Recipe", expanded=(match.rank == 1)):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📝 Ingredients:**")
            for ing in recipe.ingredients[:10]:
                st.markdown(f"- {ing}")
            if len(recipe.ingredients) > 10:
                st.caption(f"... and {len(recipe.ingredients) - 10} more")
        
        with col2:
            st.markdown("**👨‍🍳 Instructions:**")
            for i, step in enumerate(recipe.instructions[:6], 1):
                st.markdown(f"{i}. {step[:150]}{'...' if len(step) > 150 else ''}")
            if len(recipe.instructions) > 6:
                st.caption(f"... and {len(recipe.instructions) - 6} more steps")
        
        if recipe.serving_suggestion and recipe.serving_suggestion not in ['serve.', 'serve']:
            st.info(f"🍴 **Serving Suggestion:** {recipe.serving_suggestion}")
        
        if recipe.tip and 'taste and adjust' not in recipe.tip.lower():
            st.success(f"💡 **Tip:** {recipe.tip}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Neural Semantic Search</h1>
    <p>GPT-2 Embeddings for True Semantic Understanding</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    model_path = st.text_input("📁 Model Path", value="./model_finetuned")
    top_k = st.slider("🔢 Number of Results", min_value=1, max_value=20, value=10)
    show_pipeline = st.checkbox("🔍 Show Processing Pipeline", value=True)
    show_vectors = st.checkbox("📊 Show Vector Representations", value=False)
    
    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. **GPT-2 encodes** your query into a 768-dim vector
    2. **Pre-computed embeddings** for all recipes
    3. **Cosine similarity** finds semantic matches
    4. **Understands meaning**, not just keywords
    
    *"poultry" ≈ "chicken"*  
    *"quick meal" ≈ "fast recipe"*
    """)
    
    st.markdown("---")
    
    # Regenerate embeddings button
    if st.button("🔄 Regenerate Embeddings", use_container_width=True):
        embeddings_path, metadata_path = get_model_artifacts_path(model_path)
        if embeddings_path.exists():
            embeddings_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        st.success("Embeddings removed! They will be recomputed on next load.")
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Load model
with st.spinner("🔄 Loading GPT-2 model..."):
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        extractor = GPT2EmbeddingExtractor(model, tokenizer, device)
        st.sidebar.success(f"✅ Model loaded on **{device}**")
        st.sidebar.info(f"📐 Embedding dim: **{extractor.embedding_dim}**")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load pre-computed embeddings from model artifacts
precomputed = load_precomputed_embeddings(model_path)

if precomputed is not None:
    recipe_embeddings, recipes = precomputed
    st.sidebar.success(f"✅ Pre-computed embeddings loaded!")
    st.sidebar.info(f"📚 **{len(recipes):,}** recipes · **{recipe_embeddings.shape}**")
else:
    # Fallback: Load recipes and compute embeddings (first time only)
    st.warning("⏳ Pre-computed embeddings not found. Computing now (this will be saved for future use)...")
    
    recipes = load_recipes_from_jsonl()
    st.sidebar.info(f"📚 **{len(recipes):,}** recipes loaded")
    
    progress_bar = st.progress(0, text="Initializing...")
    
    def update_progress(progress, text):
        progress_bar.progress(progress, text=text)
    
    recipe_embeddings = compute_embeddings_on_demand(recipes, extractor, model_path, update_progress)
    progress_bar.empty()
    st.success("✅ Embeddings computed and saved to model artifacts!")
    st.rerun()  # Reload to use cached version

# Welcome message
st.markdown("""
<div style="background-color: #e8f5e9; border-left: 4px solid #11998e; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
    <strong>🧠 Neural Semantic Search Active!</strong><br>
    This uses GPT-2 embeddings to understand meaning, not just keywords.<br>
    Try: <em>"something with poultry and red sauce"</em> → matches "Chicken Tikka Masala"
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if query := st.chat_input("Describe what you want to cook (semantic search)..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    
    with st.chat_message("assistant", avatar="🧠"):
        
        # =====================================================================
        # STEP 1: ENCODE QUERY
        # =====================================================================
        start_time = time.time()
        
        if show_pipeline:
            st.markdown("### 🧠 Neural Processing Pipeline")
            render_pipeline_step(
                "embedding",
                "Step 1: Query Embedding",
                f"Encoding '{query[:50]}{'...' if len(query) > 50 else ''}' into 768-dimensional vector"
            )
        
        query_embedding = extractor.get_embedding(query)
        embed_time = time.time() - start_time
        
        if show_vectors:
            st.markdown("**Query Vector:**")
            render_vector_preview(query_embedding)
        
        if show_pipeline:
            st.caption(f"⏱️ Embedding time: {embed_time*1000:.1f}ms")
        
        # =====================================================================
        # STEP 2: SIMILARITY COMPUTATION
        # =====================================================================
        if show_pipeline:
            render_pipeline_step(
                "similarity",
                "Step 2: Semantic Similarity",
                f"Computing cosine similarity against {len(recipes):,} recipe embeddings"
            )
        
        search_start = time.time()
        results = semantic_search(
            query=query,
            query_embedding=query_embedding,
            recipe_embeddings=recipe_embeddings,
            recipes=recipes,
            top_k=top_k
        )
        search_time = time.time() - search_start
        
        if show_pipeline:
            st.caption(f"⏱️ Search time: {search_time*1000:.1f}ms")
        
        # =====================================================================
        # STEP 3: RANKING & RESULTS
        # =====================================================================
        if show_pipeline:
            render_pipeline_step(
                "ranking",
                "Step 3: Ranked Results",
                f"Top {len(results)} matches sorted by semantic similarity"
            )
        
        st.markdown("---")
        
        # Show similarity distribution
        if results:
            similarities = [r.similarity * 100 for r in results]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Best Match", f"{similarities[0]:.1f}%")
            with col2:
                st.metric("📊 Average", f"{np.mean(similarities):.1f}%")
            with col3:
                st.metric("⏱️ Total Time", f"{(embed_time + search_time)*1000:.0f}ms")
        
        st.markdown(f"### 🏆 Top {len(results)} Semantic Matches")
        
        if results:
            for match in results:
                render_recipe_card(match)
        else:
            st.warning("No matches found. Try a different query.")
        
        # Save response summary
        response_summary = f"Found **{len(results)}** semantically similar recipes."
        if results:
            response_summary += f"\n\nTop match: **{results[0].recipe.title}** ({results[0].similarity*100:.1f}% similarity)"
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_summary
        })

# Footer
st.markdown("""
<div class="footer">
    <p>🧠 <strong>Neural Semantic Search</strong> — Powered by GPT-2 Embeddings</p>
    <p>Understanding meaning through 768-dimensional vector representations</p>
</div>
""", unsafe_allow_html=True)
