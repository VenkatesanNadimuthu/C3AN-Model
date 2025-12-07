# 📚 Guide: Preparing a Pre-training Dataset for GPT-2

This guide explains how to prepare a high-quality dataset for pre-training a GPT-2 language model from scratch.

---

## 🎯 Goal of Pre-training

Pre-training teaches the model:
- **Language patterns** specific to your domain
- **Vocabulary** and terminology
- **Structure** and formatting conventions
- **Statistical relationships** between tokens

The model learns to predict the next token given previous tokens (causal language modeling).

---

## 📋 Requirements for a Good Pre-training Dataset

### 1. Sufficient Volume

| Dataset Size | Recommendation |
|--------------|----------------|
| < 1,000 samples | ❌ Too small - model will overfit |
| 1,000 - 5,000 samples | ⚠️ Minimum viable - expect limited generalization |
| 5,000 - 50,000 samples | ✅ Good - reasonable diversity |
| 50,000+ samples | ✅ Excellent - strong generalization |

**Rule of thumb**: At least 10-50 examples per concept you want the model to learn.

### 2. Consistent Structure

The model learns patterns from repetition. Use **consistent formatting** across all samples:

```
✅ GOOD: Same structure every time
[BOS]
**Title:** Recipe Name
**Ingredients:**
- item1
- item2
**Instructions:**
1. step1
2. step2
[EOS]

❌ BAD: Inconsistent structure
Sample 1: "Here's a recipe for cake. You need flour, sugar..."
Sample 2: "Ingredients: flour, sugar. Steps: Mix them."
Sample 3: "CAKE RECIPE - Flour - Sugar - Mix together"
```

### 3. Clean, High-Quality Data

| Issue | Impact | Solution |
|-------|--------|----------|
| Typos/errors | Model learns mistakes | Spell-check and proofread |
| Incomplete entries | Model generates incomplete output | Filter or complete entries |
| Duplicates | Overfit on repeated content | Deduplicate dataset |
| Irrelevant content | Dilutes domain knowledge | Filter aggressively |

### 4. Representative Coverage

Ensure your dataset covers:
- All categories/types you want to generate
- Edge cases and variations
- Different lengths and complexities

---

## 🔧 Step-by-Step Preparation Process

### Step 1: Collect Raw Data

```python
# Example: Scraping or loading from various sources
raw_data = []

# From CSV
import pandas as pd
df = pd.read_csv("recipes.csv")
raw_data.extend(df.to_dict('records'))

# From JSON
import json
with open("recipes.json") as f:
    raw_data.extend(json.load(f))

# From text files
with open("recipes.txt") as f:
    for line in f:
        raw_data.append({"text": line.strip()})
```

### Step 2: Clean the Data

```python
import re

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common issues
    text = text.strip()
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text

cleaned_data = [clean_text(item["text"]) for item in raw_data]
```

### Step 3: Structure the Data

Transform raw data into a consistent structured format:

```python
def structure_recipe(raw_recipe: dict) -> str:
    """Convert raw recipe dict to structured format."""
    
    structured = "[BOS]\n"
    structured += f"**Title:** {raw_recipe['name']}\n\n"
    
    # Add metadata if available
    if 'cuisine' in raw_recipe:
        structured += f"**Cuisine:** {raw_recipe['cuisine']}"
        if 'diet' in raw_recipe:
            structured += f" | **Diet:** {raw_recipe['diet']}"
        if 'time' in raw_recipe:
            structured += f" | **Time:** {raw_recipe['time']}"
        structured += "\n\n"
    
    # Add ingredients as bulleted list
    structured += "**Ingredients:**\n"
    for ing in raw_recipe['ingredients']:
        structured += f"- {ing}\n"
    
    # Add instructions as numbered list
    structured += "\n**Instructions:**\n"
    for i, step in enumerate(raw_recipe['instructions'], 1):
        structured += f"{i}. {step}\n"
    
    # Add serving suggestion if available
    if 'serving' in raw_recipe:
        structured += f"\n**Serving Suggestion:** {raw_recipe['serving']}\n"
    
    structured += "[EOS]"
    
    return structured
```

### Step 4: Add Special Tokens

Special tokens help the model understand boundaries:

| Token | Purpose | When to Use |
|-------|---------|-------------|
| `[BOS]` | Beginning of sequence | Start of every sample |
| `[EOS]` | End of sequence | End of every sample |
| `[PAD]` | Padding | Fill shorter sequences |
| `[UNK]` | Unknown token | Rare/unseen characters |

```python
def add_special_tokens(text: str) -> str:
    """Ensure special tokens are present."""
    if not text.startswith("[BOS]"):
        text = "[BOS]\n" + text
    if not text.endswith("[EOS]"):
        text = text + "\n[EOS]"
    return text
```

### Step 5: Validate and Filter

```python
def validate_sample(text: str, min_length: int = 100, max_length: int = 10000) -> bool:
    """Check if sample is valid for training."""
    
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Check for required sections (for recipes)
    required = ["**Title:**", "**Ingredients:**", "**Instructions:**"]
    for req in required:
        if req not in text:
            return False
    
    # Check for special tokens
    if "[BOS]" not in text or "[EOS]" not in text:
        return False
    
    return True

# Filter invalid samples
valid_data = [s for s in structured_data if validate_sample(s)]
print(f"Valid samples: {len(valid_data)} / {len(structured_data)}")
```

### Step 6: Format for Training

For tokenizer training, keep one sample per line:

```python
def format_for_training(text: str) -> str:
    """Convert multi-line text to single line with markers."""
    # Replace actual newlines with marker
    return text.replace('\n', ' \\n ')

# Write to file
with open("pretrain_dataset.txt", "w", encoding="utf-8") as f:
    for sample in valid_data:
        line = format_for_training(sample)
        f.write(line + "\n")
```

---

## 📊 Dataset Quality Checklist

Before training, verify:

- [ ] **Sufficient size**: At least 1,000+ samples (ideally 5,000+)
- [ ] **Consistent format**: All samples follow the same structure
- [ ] **Special tokens**: Every sample has `[BOS]` and `[EOS]`
- [ ] **No duplicates**: Deduplicated dataset
- [ ] **Clean text**: No HTML, encoding issues, or garbage characters
- [ ] **Balanced coverage**: Representative of all categories
- [ ] **Appropriate length**: Samples fit within model's context window (3000 tokens)
- [ ] **Valid structure**: All required sections present

---

## 🔍 Example: Complete Workflow

```python
#!/usr/bin/env python3
"""Complete pre-training dataset preparation workflow."""

import json
import re
from pathlib import Path
from typing import List, Dict

def load_raw_data(filepath: str) -> List[Dict]:
    """Load raw recipe data from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<[^>]+>', '', text)
    return text

def structure_recipe(recipe: Dict) -> str:
    """Convert to structured format."""
    parts = ["[BOS]"]
    parts.append(f"**Title:** {clean_text(recipe['name'])}")
    parts.append("")
    
    # Metadata
    meta = []
    if recipe.get('cuisine'):
        meta.append(f"**Cuisine:** {recipe['cuisine']}")
    if recipe.get('diet'):
        meta.append(f"**Diet:** {recipe['diet']}")
    if recipe.get('prep_time') and recipe.get('cook_time'):
        meta.append(f"**Time:** {recipe['prep_time']} prep, {recipe['cook_time']} cook")
    if meta:
        parts.append(" | ".join(meta))
        parts.append("")
    
    # Ingredients
    parts.append("**Ingredients:**")
    for ing in recipe.get('ingredients', []):
        parts.append(f"- {clean_text(ing)}")
    parts.append("")
    
    # Instructions
    parts.append("**Instructions:**")
    for i, step in enumerate(recipe.get('instructions', []), 1):
        parts.append(f"{i}. {clean_text(step)}")
    parts.append("")
    
    # Serving
    if recipe.get('serving_suggestion'):
        parts.append(f"**Serving Suggestion:** {clean_text(recipe['serving_suggestion'])}")
    
    parts.append("[EOS]")
    
    return "\n".join(parts)

def validate(text: str) -> bool:
    """Validate structured recipe."""
    required = ["[BOS]", "[EOS]", "**Title:**", "**Ingredients:**", "**Instructions:**"]
    return all(r in text for r in required) and len(text) >= 100

def main():
    # Load
    raw_data = load_raw_data("raw_recipes.json")
    print(f"Loaded {len(raw_data)} raw recipes")
    
    # Process
    structured = [structure_recipe(r) for r in raw_data]
    valid = [s for s in structured if validate(s)]
    print(f"Valid recipes: {len(valid)}")
    
    # Deduplicate
    unique = list(set(valid))
    print(f"Unique recipes: {len(unique)}")
    
    # Save
    with open("pretrain_dataset.txt", "w", encoding="utf-8") as f:
        for recipe in unique:
            line = recipe.replace('\n', ' \\n ')
            f.write(line + "\n")
    
    print("✓ Dataset saved to pretrain_dataset.txt")

if __name__ == "__main__":
    main()
```

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Why It's Bad | Solution |
|---------|--------------|----------|
| Inconsistent formatting | Model learns chaos | Use templates |
| Missing special tokens | No clear boundaries | Always add `[BOS]`/`[EOS]` |
| Too much variety in structure | Model can't learn patterns | Standardize format |
| Including test data | Data leakage | Separate train/test |
| Ignoring encoding | Unicode errors | Use UTF-8 everywhere |
| Very long samples | Truncation issues | Keep under context limit |

---

## 📈 Scaling Tips

For larger datasets:

1. **Use streaming**: Don't load everything into memory
2. **Parallelize**: Use multiprocessing for data cleaning
3. **Validate incrementally**: Check samples as you process
4. **Use checksums**: Detect duplicates efficiently with hashing

```python
import hashlib
from multiprocessing import Pool

def hash_sample(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Efficient deduplication
seen_hashes = set()
unique_samples = []
for sample in samples:
    h = hash_sample(sample)
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_samples.append(sample)
```

---

## 🎓 Summary

1. **Collect** diverse, high-quality raw data
2. **Clean** thoroughly (whitespace, encoding, HTML)
3. **Structure** consistently with explicit markers
4. **Add** special tokens (`[BOS]`, `[EOS]`)
5. **Validate** every sample meets requirements
6. **Format** for training (one sample per line)

The quality of your pre-training dataset directly determines the quality of your model's output!
