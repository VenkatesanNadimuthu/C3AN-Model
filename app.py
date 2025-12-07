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
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
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
def generate_recipe(
    instruction: str,
    model,
    tokenizer,
    device,
    temperature: float = 0.8,
    max_new_tokens: int = 300,
    top_k: int = 50,
    top_p: float = 0.92,
) -> tuple[str, str]:
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
        Tuple of (formatted_prompt, generated_response)
    """
    # Format as instruction prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return prompt, response.strip()


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
                    raw_prompt, response = generate_recipe(
                        instruction=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        max_new_tokens=max_length,
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show raw prompt if debug is enabled
                    if show_raw_prompt:
                        with st.expander("🔍 Debug: Raw Prompt"):
                            st.code(raw_prompt, language="text")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "raw_prompt": raw_prompt
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating recipe: {str(e)}"
                    st.error(error_msg)
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
