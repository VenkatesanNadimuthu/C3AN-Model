# ============================================================================
# 🍳 GOURMET GPT: STREAMLIT DEPLOYMENT ON GOOGLE COLAB
# ============================================================================
# This script contains all the commands to deploy your Recipe Chatbot
# Copy these cells into Google Colab and run them in order.
# ============================================================================

# ============================================================================
# CELL 1: INSTALL REQUIREMENTS
# ============================================================================
# Run this cell first to install all dependencies

"""
!pip install -q streamlit transformers torch pyngrok
"""

# ============================================================================
# CELL 2: UPLOAD/VERIFY MODEL
# ============================================================================
# Make sure your model is in ./model_finetuned/
# If uploading from Google Drive:

"""
# Option A: Mount Google Drive (if model is stored there)
from google.colab import drive
drive.mount('/content/drive')

# Copy model from Drive to Colab
!cp -r "/content/drive/MyDrive/your_model_folder" ./model_finetuned

# Option B: Upload the model zip file and extract
# from google.colab import files
# uploaded = files.upload()  # Upload your model zip
# !unzip gpt2_recipe_instruct_model.zip -d ./model_finetuned
"""

# Verify model files exist
"""
!ls -la ./model_finetuned/
"""

# ============================================================================
# CELL 3: CREATE app.py
# ============================================================================
# This creates the Streamlit application file

APP_CODE = '''
"""
🍳 Gourmet GPT: Bespoke Recipe Assistant
A professional Streamlit chatbot for recipe generation using fine-tuned GPT-2.
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Page config
st.set_page_config(
    page_title="Gourmet GPT: Bespoke Recipe Assistant",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    .info-box {
        background-color: #e8f4f8; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
    .footer {
        text-align: center; padding: 1rem; color: #6c757d;
        font-size: 0.85rem; margin-top: 2rem; border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer(model_path: str = "./model_finetuned"):
    """Load model and tokenizer (cached to prevent Colab crashes)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_recipe(instruction, model, tokenizer, device, temperature=0.8, max_new_tokens=300):
    """Generate a recipe from user instruction."""
    prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt, response.strip()


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    model_path = st.text_input("📁 Model Path", value="./model_finetuned")
    
    st.markdown("---")
    st.markdown("### 🎛️ Generation Parameters")
    
    temperature = st.slider("🌡️ Creativity (Temperature)", 0.1, 1.0, 0.8, 0.05)
    max_length = st.slider("📏 Recipe Length (Max Tokens)", 100, 1000, 300, 50)
    
    st.markdown("---")
    show_raw_prompt = st.checkbox("📝 Show Raw Prompt", value=False)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        st.success(f"✅ Model loaded on **{device}**")
        st.info(f"📈 Parameters: **{sum(p.numel() for p in model.parameters()):,}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        model, tokenizer, device = None, None, None


# Header
st.markdown("""
<div class="main-header">
    <h1>🍳 Gourmet GPT</h1>
    <p>Your Bespoke Recipe Assistant — Powered by Fine-Tuned GPT-2</p>
</div>
""", unsafe_allow_html=True)

# Welcome
if "welcomed" not in st.session_state:
    st.markdown("""
    <div class="info-box">
        <strong>👋 Welcome to Gourmet GPT!</strong><br>
        Tell me what ingredients you have or what dish you'd like — I'll craft a recipe for you!
    </div>
    """, unsafe_allow_html=True)
    st.session_state.welcomed = True

# Examples
with st.expander("💡 **Example Prompts**"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- \\"Give me a recipe for chocolate cake\\"\\n- \\"I have chicken, rice, and bell peppers\\"")
    with col2:
        st.markdown("- \\"Create a vegetarian pasta dish\\"\\n- \\"Quick breakfast with eggs\\"")

st.markdown("---")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🍳"):
        st.markdown(message["content"])
        if show_raw_prompt and message["role"] == "assistant" and "raw_prompt" in message:
            with st.expander("🔍 Debug: Raw Prompt"):
                st.code(message["raw_prompt"])

# Chat input
if prompt := st.chat_input("What recipe would you like today?"):
    if model is None:
        st.error("⚠️ Model not loaded. Check the model path.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🍳"):
            with st.spinner("👨‍🍳 Crafting your recipe..."):
                try:
                    raw_prompt, response = generate_recipe(
                        prompt, model, tokenizer, device, temperature, max_length
                    )
                    st.markdown(response)
                    if show_raw_prompt:
                        with st.expander("🔍 Debug: Raw Prompt"):
                            st.code(raw_prompt)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response, "raw_prompt": raw_prompt
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.markdown("""
<div class="footer">
    <p>🍳 <strong>Gourmet GPT</strong> — Fine-tuned on curated recipe data</p>
    <p>Built with ❤️ using Streamlit & HuggingFace Transformers</p>
</div>
""", unsafe_allow_html=True)
'''

# Write app.py
"""
with open("app.py", "w") as f:
    f.write(APP_CODE)
    
print("✅ app.py created successfully!")
!ls -la app.py
"""

# ============================================================================
# CELL 4: SETUP NGROK FOR PUBLIC URL
# ============================================================================
# Replace YOUR_AUTHTOKEN with your actual ngrok auth token
# Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken

"""
from pyngrok import ngrok

# ⚠️ IMPORTANT: Replace with your actual ngrok auth token
NGROK_AUTH_TOKEN = "YOUR_AUTHTOKEN"

# Authenticate ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Kill any existing tunnels
ngrok.kill()

print("✅ ngrok authenticated!")
"""

# ============================================================================
# CELL 5: LAUNCH STREAMLIT WITH NGROK TUNNEL
# ============================================================================
# This runs Streamlit in the background and creates a public URL

"""
import subprocess
import time
from pyngrok import ngrok

# Start Streamlit in background
streamlit_process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
public_url = ngrok.connect(8501)

print("=" * 70)
print("🍳 GOURMET GPT IS LIVE!")
print("=" * 70)
print(f"\\n🌐 PUBLIC URL: {public_url}")
print(f"\\n📱 Share this link with anyone to access your Recipe Chatbot!")
print("\\n⚠️  Keep this Colab tab open to maintain the connection.")
print("=" * 70)
"""

# ============================================================================
# CELL 6: ALTERNATIVE - RUN WITHOUT NGROK (localtunnel)
# ============================================================================
# If ngrok doesn't work, try localtunnel instead

"""
# Install localtunnel
!npm install -g localtunnel

# Run Streamlit in background
!streamlit run app.py --server.port 8501 --server.headless true &

# Wait and create tunnel
import time
time.sleep(5)
!lt --port 8501
"""

# ============================================================================
# CELL 7: STOP THE SERVER (Run when done)
# ============================================================================

"""
# To stop the server:
from pyngrok import ngrok
ngrok.kill()

# Or kill streamlit process
!pkill -f streamlit

print("✅ Server stopped!")
"""
