"""
🍳 Gourmet GPT: Bespoke Recipe Assistant
A professional Streamlit chatbot for recipe generation using fine-tuned GPT-2.

Author: AI Research Engineer
Date: 2025-12-06
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gourmet GPT: Bespoke Recipe Assistant",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS FOR POLISHED LOOK
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Chat message styling - Enhanced for structured responses */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Recipe card styling - ChatGPT/Perplexity style */
    .recipe-response {
        background: linear-gradient(to bottom right, #fafafa, #f5f5f5);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .recipe-response h3 {
        color: #1a1a1a;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .recipe-section {
        margin: 1rem 0;
        padding: 0.75rem;
        background: white;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    .recipe-section h4 {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recipe-section ul {
        margin: 0.5rem 0 0 1rem;
        padding: 0;
    }
    
    .recipe-section li {
        margin: 0.3rem 0;
        line-height: 1.5;
        color: #333;
    }
    
    .recipe-section ol {
        margin: 0.5rem 0 0 1rem;
        padding: 0;
    }
    
    .recipe-section ol li {
        margin: 0.5rem 0;
        padding-left: 0.25rem;
    }
    
    /* Tip/Serving suggestion styling */
    .recipe-tip {
        background: linear-gradient(135deg, #fff9e6, #fff5d6);
        border-left: 3px solid #f0ad4e;
        margin-top: 1rem;
    }
    
    .recipe-tip h4 {
        color: #c08b30;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
    }
    
    /* Info box styling */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================
@st.cache_resource
def load_model_and_tokenizer(model_path: str = "./model_finetuned"):
    """
    Load the fine-tuned GPT-2 model and tokenizer.
    Uses st.cache_resource to load only once and prevent Colab crashes.
    
    Args:
        model_path: Path to the fine-tuned model directory
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # Load model and move to device
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


# ============================================================================
# GENERATION FUNCTION
# ============================================================================
def format_structured_response(response: str) -> tuple[str, bool]:
    """
    Format the structured response for ChatGPT/Perplexity style display.
    Converts markdown-style formatting to rich HTML cards.
    
    Handles both clean markers (**Title:**) and spaced markers (** Title :**)
    which can occur due to tokenization artifacts.
    
    Returns:
        Tuple of (formatted_html, is_structured)
    """
    import re
    
    # Clean up any artifacts
    response = response.strip()
    
    # Remove any trailing [EOS] tokens
    response = re.sub(r'\[EOS\]\s*$', '', response)
    
    # Normalize spaced markers: "** Title :**" -> "**Title:**"
    # This handles tokenization artifacts from the model
    response = re.sub(r'\*\*\s*Title\s*:\s*\*\*', '**Title:**', response)
    response = re.sub(r'\*\*\s*Ingredients\s*:\s*\*\*', '**Ingredients:**', response)
    response = re.sub(r'\*\*\s*Instructions\s*:\s*\*\*', '**Instructions:**', response)
    response = re.sub(r'\*\*\s*Serving\s*Suggestion\s*:\s*\*\*', '**Serving Suggestion:**', response)
    response = re.sub(r'\*\*\s*Tip\s*:\s*\*\*', '**Tip:**', response)
    response = re.sub(r'\*\*\s*Cuisine\s*:\s*\*\*', '**Cuisine:**', response)
    response = re.sub(r'\*\*\s*Diet\s*:\s*\*\*', '**Diet:**', response)
    response = re.sub(r'\*\*\s*Time\s*:\s*\*\*', '**Time:**', response)
    
    # Also handle "Inst ruction" -> "Instruction" type spacing issues
    response = re.sub(r'Inst\s*ruction', 'Instruction', response)
    response = re.sub(r'Res\s*pon\s*se', 'Response', response)
    response = re.sub(r'Ste\s*p', 'Step', response)
    
    # If response doesn't have structured markers, return as plain markdown
    if '**Title:**' not in response and '**Ingredients:**' not in response:
        return response, False
    
    # Parse structured sections
    html_parts = ['<div class="recipe-response">']
    
    # Extract Title
    title_match = re.search(r'\*\*Title:\*\*\s*([^\n*]+)', response)
    if title_match:
        title = title_match.group(1).strip()
        html_parts.append(f'<h3>🍽️ {title}</h3>')
    
    # Extract Cuisine/Diet/Time (optional metadata)
    for meta, emoji in [('Cuisine', '🌍'), ('Diet', '🥗'), ('Time', '⏱️')]:
        meta_match = re.search(rf'\*\*{meta}:\*\*\s*([^\n*]+)', response)
        if meta_match:
            html_parts.append(f'<p><strong>{emoji} {meta}:</strong> {meta_match.group(1).strip()}</p>')
    
    # Extract Ingredients section
    ingredients_match = re.search(
        r'\*\*Ingredients:\*\*\s*\n?([\s\S]*?)(?=\*\*Instructions:|\*\*Serving|\*\*Tip:|$)',
        response
    )
    if ingredients_match:
        ingredients_text = ingredients_match.group(1).strip()
        # Parse bullet points (handle both "- item" and "- item")
        items = re.findall(r'-\s*(.+?)(?=\s*-\s|\s*\*\*|$)', ingredients_text)
        if items:
            html_parts.append('<div class="recipe-section">')
            html_parts.append('<h4>📝 Ingredients</h4>')
            html_parts.append('<ul>')
            for item in items:
                clean_item = item.strip()
                if clean_item and len(clean_item) > 1:
                    html_parts.append(f'<li>{clean_item}</li>')
            html_parts.append('</ul>')
            html_parts.append('</div>')
    
    # Extract Instructions section
    instructions_match = re.search(
        r'\*\*Instructions:\*\*\s*\n?([\s\S]*?)(?=\*\*Serving|\*\*Tip:|$)',
        response
    )
    if instructions_match:
        instructions_text = instructions_match.group(1).strip()
        # Parse numbered steps (handle "1 ." and "1." formats)
        steps = re.findall(r'\d+\s*\.\s*(.+?)(?=\d+\s*\.|$)', instructions_text, re.DOTALL)
        if steps:
            html_parts.append('<div class="recipe-section">')
            html_parts.append('<h4>👨‍🍳 Instructions</h4>')
            html_parts.append('<ol>')
            for step in steps:
                clean_step = step.strip()
                if clean_step and len(clean_step) > 1:
                    html_parts.append(f'<li>{clean_step}</li>')
            html_parts.append('</ol>')
            html_parts.append('</div>')
    
    # Extract Serving Suggestion (optional)
    serving_match = re.search(r'\*\*Serving Suggestion:\*\*\s*([^\n*]+)', response)
    if serving_match:
        html_parts.append('<div class="recipe-section recipe-tip">')
        html_parts.append('<h4>🍴 Serving Suggestion</h4>')
        html_parts.append(f'<p>{serving_match.group(1).strip()}</p>')
        html_parts.append('</div>')
    
    # Extract Tip (optional)
    tip_match = re.search(r'\*\*Tip:\*\*\s*([^\n*]+)', response)
    if tip_match:
        html_parts.append('<div class="recipe-section recipe-tip">')
        html_parts.append('<h4>💡 Chef\'s Tip</h4>')
        html_parts.append(f'<p>{tip_match.group(1).strip()}</p>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    # Only return HTML if we actually parsed structured content
    if len(html_parts) > 2:  # More than just opening/closing div
        return '\n'.join(html_parts), True
    else:
        return response, False


def validate_structured_response(response: str) -> dict:
    """
    Validate that the generated response follows structured format.
    
    Args:
        response: Generated recipe text (before formatting)
        
    Returns:
        dict with validation results
    """
    result = {
        "is_structured": False,
        "has_title": "**Title:**" in response or "🍽️ **Recipe:**" in response,
        "has_ingredients": "**Ingredients:**" in response or "📝 **Ingredients:**" in response,
        "has_instructions": "**Instructions:**" in response or "👨‍🍳 **Instructions:**" in response,
        "has_serving": "**Serving Suggestion:**" in response or "🍴 **Serving Suggestion:**" in response,
        "has_bullets": "- " in response,
        "has_numbers": any(f"{i}." in response for i in range(1, 10)),
    }
    
    # Fully structured if has all three required markers
    result["is_structured"] = (
        result["has_title"] and 
        result["has_ingredients"] and 
        result["has_instructions"]
    )
    
    return result


def generate_recipe(
    instruction: str,
    model,
    tokenizer,
    device,
    temperature: float = 0.8,
    max_new_tokens: int = 300,
    top_k: int = 50,
    top_p: float = 0.92,
) -> tuple[str, str, dict, bool]:
    """
    Generate a recipe from a user instruction.
    
    Args:
        instruction: User's request
        model: The GPT-2 model
        tokenizer: The tokenizer
        device: torch device
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        
    Returns:
        Tuple of (formatted_prompt, generated_response, validation_result, is_html)
    
    Note: When trained on structured datasets, the model outputs:
        **Title:** Recipe Name
        **Ingredients:** (bulleted list)
        **Instructions:** (numbered steps)
        **Serving Suggestion:** ...
    """
    # Format as instruction prompt (for fine-tuned model)
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
    # Clear CUDA cache if using GPU to prevent memory issues
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # Force at least some output
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    
    # Decode only the generated part (after prompt)
    generated_ids = outputs[0][prompt_length:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up response - remove any repeated instruction pattern
    if "### Instruction:" in raw_response:
        raw_response = raw_response.split("### Instruction:")[0]
    
    # Validate structured format BEFORE formatting
    validation = validate_structured_response(raw_response)
    
    # If response is empty, return a helpful message
    if not raw_response.strip():
        response = "(Model generated empty response. Try rephrasing your request or adjusting temperature.)"
        validation = {"is_structured": False, "has_title": False, "has_ingredients": False, "has_instructions": False}
        is_html = False
    else:
        # Format structured response for ChatGPT/Perplexity style display
        response, is_html = format_structured_response(raw_response)
    
    return prompt, response.strip(), validation, is_html


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Model path input
    model_path = st.text_input(
        "📁 Model Path",
        value="./model_finetuned",
        help="Path to your fine-tuned model directory"
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Generation Parameters")
    
    # Temperature slider
    temperature = st.slider(
        "🌡️ Creativity (Temperature)",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Higher values = more creative, lower = more focused"
    )
    
    # Max length slider
    max_length = st.slider(
        "📏 Recipe Length (Max Tokens)",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Debug Options")
    
    # Debug checkbox
    show_raw_prompt = st.checkbox(
        "📝 Show Raw Prompt",
        value=False,
        help="Display the formatted prompt sent to the model"
    )
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Model info
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        st.success(f"✅ Model loaded on **{device}**")
        param_count = sum(p.numel() for p in model.parameters())
        st.info(f"📈 Parameters: **{param_count:,}**")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        model, tokenizer, device = None, None, None


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🍳 Gourmet GPT</h1>
    <p>Your Bespoke Recipe Assistant — Powered by Fine-Tuned GPT-2</p>
</div>
""", unsafe_allow_html=True)

# Welcome message
if "welcomed" not in st.session_state:
    st.markdown("""
    <div class="info-box">
        <strong>👋 Welcome to Gourmet GPT!</strong><br>
        I'm your personal recipe assistant. Tell me what ingredients you have, 
        what cuisine you're craving, or what dish you'd like to make — and I'll 
        craft a custom recipe just for you!
    </div>
    """, unsafe_allow_html=True)
    st.session_state.welcomed = True

# Example prompts
with st.expander("💡 **Example Prompts** — Click to expand"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - "Give me a recipe for chocolate cake"
        - "I have chicken, rice, and bell peppers"
        - "Make me a quick breakfast with eggs"
        """)
    with col2:
        st.markdown("""
        - "Create a vegetarian pasta dish"
        - "I want something spicy with shrimp"
        - "Healthy dinner recipe under 500 calories"
        """)

st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🍳"):
        # Render HTML for structured responses, markdown for plain text
        if message.get("is_html", False):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])
        
        # Show raw prompt if enabled and it's an assistant message with debug info
        if show_raw_prompt and message["role"] == "assistant" and "raw_prompt" in message:
            with st.expander("🔍 Debug: Raw Prompt"):
                st.code(message["raw_prompt"], language="text")

# Chat input
if prompt := st.chat_input("What recipe would you like today?", key="chat_input"):
    
    # Check if model is loaded
    if model is None:
        st.error("⚠️ Model not loaded. Please check the model path in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🍳"):
            with st.spinner("👨‍🍳 Crafting your recipe..."):
                try:
                    # Force garbage collection before generation
                    import gc
                    gc.collect()
                    
                    raw_prompt, response, validation, is_html = generate_recipe(
                        instruction=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        max_new_tokens=max_length,
                    )
                    
                    # Display response - use HTML for structured, markdown for plain
                    if is_html:
                        st.markdown(response, unsafe_allow_html=True)
                    else:
                        st.markdown(response)
                    
                    # Show structured format validation status
                    if validation["is_structured"]:
                        st.caption("✅ Structured format: Title, Ingredients, Instructions")
                    else:
                        markers = []
                        if validation.get("has_title"): markers.append("Title")
                        if validation.get("has_ingredients"): markers.append("Ingredients")
                        if validation.get("has_instructions"): markers.append("Instructions")
                        if markers:
                            st.caption(f"⚠️ Partial format: {', '.join(markers)}")
                        else:
                            st.caption("ℹ️ Freeform response (no structured markers)")
                    
                    # Show raw prompt if debug is enabled
                    if show_raw_prompt:
                        with st.expander("🔍 Debug: Raw Prompt"):
                            st.code(raw_prompt, language="text")
                        with st.expander("🔍 Debug: Validation Details"):
                            st.json(validation)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "raw_prompt": raw_prompt,
                        "validation": validation,
                        "is_html": is_html
                    })
                    
                    # Post-generation cleanup
                    gc.collect()
                    
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    error_msg = f"❌ Error generating recipe: {str(e)}"
                    st.error(error_msg)
                    if show_raw_prompt:
                        with st.expander("🔍 Debug: Error Traceback"):
                            st.code(error_trace, language="text")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.markdown("""
<div class="footer">
    <p>🍳 <strong>Gourmet GPT</strong> — Fine-tuned on curated recipe data</p>
    <p>Built with ❤️ using Streamlit & HuggingFace Transformers</p>
</div>
""", unsafe_allow_html=True)
