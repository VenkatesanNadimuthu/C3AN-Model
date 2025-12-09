"""
Pre-compute and save GPT-2 embeddings for all recipes.
Run this once to generate embeddings as a model artifact.

Usage:
    python precompute_embeddings.py

Output:
    model_finetuned/recipe_embeddings.npz - Embeddings + metadata
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(model_path: str):
    """Load GPT-2 model and tokenizer."""
    print(f"📂 Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer, device


def parse_recipe_response(response: str) -> dict | None:
    """Parse structured recipe response."""
    import re
    
    recipe = {
        'title': '',
        'ingredients': [],
        'instructions': [],
        'serving_suggestion': '',
        'tip': '',
    }
    
    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n|$)', response)
    if title_match:
        recipe['title'] = title_match.group(1).strip()
    else:
        return None
    
    ingredients_match = re.search(
        r'\*\*Ingredients:\*\*\s*\n?([\s\S]*?)(?=\*\*Instructions:|\*\*Serving|\*\*Tip:|$)',
        response
    )
    if ingredients_match:
        items = re.findall(r'-\s*(.+?)(?=\s*-\s|\s*\*\*|$)', ingredients_match.group(1))
        recipe['ingredients'] = [item.strip() for item in items if item.strip()]
    
    instructions_match = re.search(
        r'\*\*Instructions:\*\*\s*\n?([\s\S]*?)(?=\*\*Serving|\*\*Tip:|$)',
        response
    )
    if instructions_match:
        steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', instructions_match.group(1), re.DOTALL)
        recipe['instructions'] = [step.strip() for step in steps if step.strip()]
    
    serving_match = re.search(r'\*\*Serving Suggestion:\*\*\s*(.+?)(?:\n|$)', response)
    if serving_match:
        recipe['serving_suggestion'] = serving_match.group(1).strip()
    
    tip_match = re.search(r'\*\*Tip:\*\*\s*(.+?)(?:\n|$)', response)
    if tip_match:
        recipe['tip'] = tip_match.group(1).strip()
    
    # Create combined text for embedding
    recipe['text'] = f"{recipe['title']}. Ingredients: {', '.join(recipe['ingredients'][:15])}. {' '.join(recipe['instructions'][:3])}"
    
    return recipe


def load_recipes(filepath: str) -> list[dict]:
    """Load recipes from JSONL file."""
    print(f"📂 Loading recipes from {filepath}...")
    recipes = []
    seen_titles = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing recipes"):
            try:
                data = json.loads(line.strip())
                response = data.get('response', '')
                recipe = parse_recipe_response(response)
                
                if recipe and recipe['title'] not in seen_titles:
                    seen_titles.add(recipe['title'])
                    recipes.append(recipe)
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Loaded {len(recipes):,} unique recipes")
    return recipes


@torch.no_grad()
def compute_embeddings(
    texts: list[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256
) -> np.ndarray:
    """
    Compute embeddings for all texts using GPT-2.
    
    Uses mean pooling of the last hidden state.
    """
    print(f"🧠 Computing embeddings for {len(texts):,} texts...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        
        # Mean pooling with attention mask
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


def save_embeddings(
    embeddings: np.ndarray,
    recipes: list[dict],
    output_path: str
):
    """
    Save embeddings and recipe metadata as model artifact.
    """
    print(f"💾 Saving embeddings to {output_path}...")
    
    # Prepare metadata
    titles = [r['title'] for r in recipes]
    texts = [r['text'] for r in recipes]
    
    # Save as compressed numpy archive
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        titles=np.array(titles, dtype=object),
        texts=np.array(texts, dtype=object)
    )
    
    # Also save full recipe data as JSON
    recipe_json_path = Path(output_path).parent / "recipes_metadata.json"
    with open(recipe_json_path, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    
    # Print stats
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ Saved embeddings: {output_path} ({file_size:.2f} MB)")
    print(f"✅ Saved metadata: {recipe_json_path}")
    print(f"📊 Embedding shape: {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute GPT-2 embeddings for recipes")
    parser.add_argument("--model-path", default="./model_finetuned", help="Path to fine-tuned GPT-2 model")
    parser.add_argument("--recipes-path", default="./Dataset/structured_recipes_finetune.jsonl", help="Path to recipes JSONL")
    parser.add_argument("--output-path", default="./model_finetuned/recipe_embeddings.npz", help="Output path for embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length for embedding")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍳 GPT-2 Recipe Embedding Pre-computation")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    # Load recipes
    recipes = load_recipes(args.recipes_path)
    
    # Extract texts for embedding
    texts = [r['text'] for r in recipes]
    
    # Compute embeddings
    embeddings = compute_embeddings(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Save as model artifact
    save_embeddings(embeddings, recipes, args.output_path)
    
    print("=" * 60)
    print("✅ Done! Embeddings saved as model artifact.")
    print(f"   Add these files to your model_finetuned/ directory:")
    print(f"   - recipe_embeddings.npz")
    print(f"   - recipes_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
