# 🎯 Guide: Preparing a Fine-tuning Dataset for Instruction Following

This guide explains how to prepare a high-quality instruction fine-tuning dataset to align your pre-trained model with user requests.

---

## 🎯 Goal of Instruction Fine-tuning

Fine-tuning teaches the model to:
- **Follow instructions** from users
- **Generate appropriate responses** in the expected format
- **Generalize** to new instructions it hasn't seen
- **Maintain** the domain knowledge from pre-training

---

## 📋 Key Differences: Pre-training vs Fine-tuning

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| **Purpose** | Learn language patterns | Learn to follow instructions |
| **Format** | Raw text with structure | Instruction-Response pairs |
| **Data size** | Large (thousands) | Smaller (hundreds to thousands) |
| **Learning rate** | Higher (5e-5) | Lower (1e-5 to 1e-6) |
| **Epochs** | More (10+) | Fewer (1-5) |
| **Goal** | Domain knowledge | Task alignment |

---

## 📊 Dataset Format: Alpaca-style JSONL

The standard format for instruction fine-tuning:

```json
{"instruction": "User's request", "response": "Model's response"}
```

### Basic Example:

```json
{"instruction": "Give me a recipe for chocolate cake.", "response": "**Title:** Chocolate Cake\n\n**Ingredients:**\n- 2 cups flour\n- 1 cup sugar\n- 1/2 cup cocoa powder\n- 2 eggs\n- 1 cup milk\n\n**Instructions:**\n1. Preheat oven to 350°F.\n2. Mix dry ingredients in a bowl.\n3. Add eggs and milk, stir until smooth.\n4. Pour into greased pan.\n5. Bake for 30 minutes.\n\n**Serving Suggestion:** Serve with vanilla ice cream."}
```

### Optional Extended Format (with input context):

```json
{"instruction": "Suggest a healthier version.", "input": "Original recipe: Fried Chicken with lots of oil", "response": "**Title:** Air-Fried Chicken\n\n**Ingredients:**\n- Chicken breast\n- Light coating of olive oil\n- Herbs and spices\n\n**Instructions:**\n1. Season chicken with herbs.\n2. Lightly coat with olive oil.\n3. Air fry at 375°F for 20 minutes.\n\n**Tip:** This version uses 90% less oil!"}
```

---

## 🔧 Step-by-Step Preparation Process

### Step 1: Define Instruction Categories

Identify the types of instructions your model should handle:

```python
INSTRUCTION_TEMPLATES = [
    # Direct requests
    "Give me a recipe for {dish}.",
    "How do I make {dish}?",
    "Share the recipe for {dish}.",
    
    # Specific queries
    "What ingredients do I need for {dish}?",
    "Walk me through making {dish}.",
    "Provide step-by-step instructions for {dish}.",
    
    # Contextual requests
    "I have {ingredients}. What can I make?",
    "Suggest a {cuisine} recipe for dinner.",
    "Give me a quick {time} minute recipe.",
    
    # Modification requests
    "Make this recipe vegetarian: {recipe}",
    "Suggest a healthier version of {dish}.",
    "How can I make {dish} without {ingredient}?",
]
```

### Step 2: Generate Instruction Variations

Diversity in instructions helps the model generalize:

```python
import random

def generate_instruction_variations(dish_name: str, n_variations: int = 3) -> list:
    """Generate diverse instructions for the same dish."""
    
    templates = [
        f"Give me a recipe for {dish_name}.",
        f"How do I make {dish_name}?",
        f"Share the complete recipe for {dish_name}.",
        f"I want to cook {dish_name}. What are the steps?",
        f"Can you help me with the {dish_name} recipe?",
        f"Walk me through making {dish_name}.",
        f"What ingredients do I need for {dish_name} and how do I prepare it?",
        f"Provide step-by-step instructions for {dish_name}.",
    ]
    
    return random.sample(templates, min(n_variations, len(templates)))
```

### Step 3: Structure the Responses

Responses should match the structured format learned during pre-training:

```python
def create_structured_response(recipe: dict) -> str:
    """Create a structured response from recipe data."""
    
    response_parts = []
    
    # Title
    response_parts.append(f"**Title:** {recipe['name']}")
    response_parts.append("")
    
    # Ingredients (bulleted)
    response_parts.append("**Ingredients:**")
    for ing in recipe['ingredients']:
        response_parts.append(f"- {ing}")
    response_parts.append("")
    
    # Instructions (numbered)
    response_parts.append("**Instructions:**")
    for i, step in enumerate(recipe['instructions'], 1):
        response_parts.append(f"{i}. {step}")
    response_parts.append("")
    
    # Serving suggestion
    if recipe.get('serving'):
        response_parts.append(f"**Serving Suggestion:** {recipe['serving']}")
    
    # Optional tip
    if recipe.get('tip'):
        response_parts.append("")
        response_parts.append(f"**Tip:** {recipe['tip']}")
    
    return "\n".join(response_parts)
```

### Step 4: Create Instruction-Response Pairs

```python
import json

def create_finetuning_dataset(recipes: list, variations_per_recipe: int = 3) -> list:
    """Create instruction-response pairs from recipe data."""
    
    dataset = []
    
    for recipe in recipes:
        # Create structured response (same for all variations)
        response = create_structured_response(recipe)
        
        # Generate instruction variations
        instructions = generate_instruction_variations(
            recipe['name'], 
            n_variations=variations_per_recipe
        )
        
        # Create pairs
        for instruction in instructions:
            dataset.append({
                "instruction": instruction,
                "response": response
            })
    
    return dataset

# Save to JSONL
def save_jsonl(data: list, filepath: str):
    """Save dataset as JSONL (one JSON object per line)."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### Step 5: Balance and Validate

```python
from collections import Counter

def analyze_dataset(dataset: list) -> dict:
    """Analyze dataset balance and quality."""
    
    stats = {
        "total_samples": len(dataset),
        "avg_instruction_length": 0,
        "avg_response_length": 0,
        "instruction_types": Counter(),
    }
    
    inst_lengths = []
    resp_lengths = []
    
    for item in dataset:
        inst_lengths.append(len(item["instruction"]))
        resp_lengths.append(len(item["response"]))
        
        # Categorize instruction type
        inst = item["instruction"].lower()
        if "how do i" in inst:
            stats["instruction_types"]["how-to"] += 1
        elif "give me" in inst or "share" in inst:
            stats["instruction_types"]["request"] += 1
        elif "what ingredients" in inst:
            stats["instruction_types"]["ingredients"] += 1
        else:
            stats["instruction_types"]["other"] += 1
    
    stats["avg_instruction_length"] = sum(inst_lengths) / len(inst_lengths)
    stats["avg_response_length"] = sum(resp_lengths) / len(resp_lengths)
    
    return stats

def validate_sample(item: dict) -> tuple[bool, str]:
    """Validate a single instruction-response pair."""
    
    # Check required fields
    if "instruction" not in item:
        return False, "Missing 'instruction' field"
    if "response" not in item:
        return False, "Missing 'response' field"
    
    # Check non-empty
    if not item["instruction"].strip():
        return False, "Empty instruction"
    if not item["response"].strip():
        return False, "Empty response"
    
    # Check response has required structure
    required = ["**Title:**", "**Ingredients:**", "**Instructions:**"]
    for req in required:
        if req not in item["response"]:
            return False, f"Response missing {req}"
    
    # Check reasonable lengths
    if len(item["instruction"]) < 10:
        return False, "Instruction too short"
    if len(item["response"]) < 50:
        return False, "Response too short"
    
    return True, "Valid"
```

---

## 📝 Instruction Design Best Practices

### 1. Be Diverse

Include many ways to ask for the same thing:

```json
{"instruction": "How do I make pasta?", "response": "..."}
{"instruction": "Give me a pasta recipe.", "response": "..."}
{"instruction": "I want to cook pasta. Help me.", "response": "..."}
{"instruction": "Share step-by-step pasta instructions.", "response": "..."}
{"instruction": "What's the recipe for pasta?", "response": "..."}
```

### 2. Be Natural

Use conversational language users would actually type:

```
✅ GOOD (Natural):
"I'm craving something Italian. Any ideas?"
"Quick dinner recipe for 2 people?"
"Help me make something with chicken and rice"

❌ BAD (Robotic):
"Generate recipe output for Italian cuisine category"
"Produce food preparation instructions"
"Execute recipe retrieval for poultry protein"
```

### 3. Include Edge Cases

Train on challenging scenarios:

```json
{"instruction": "I only have eggs and cheese. What can I make?", "response": "..."}
{"instruction": "Give me a 10-minute recipe.", "response": "..."}
{"instruction": "Vegan version of butter chicken?", "response": "..."}
{"instruction": "Recipe for kids who hate vegetables?", "response": "..."}
```

### 4. Maintain Response Consistency

All responses should follow the exact same structure:

```
**Title:** [Name]

**Ingredients:**
- [item 1]
- [item 2]

**Instructions:**
1. [step 1]
2. [step 2]

**Serving Suggestion:** [suggestion]
```

---

## 📊 Dataset Quality Checklist

Before fine-tuning, verify:

- [ ] **Sufficient size**: At least 500+ instruction-response pairs
- [ ] **Diverse instructions**: Multiple phrasings per concept
- [ ] **Consistent responses**: All follow the same structure
- [ ] **Natural language**: Instructions sound like real users
- [ ] **Balanced categories**: Cover all instruction types
- [ ] **Valid JSON**: Every line parses correctly
- [ ] **No duplicates**: Deduplicated pairs
- [ ] **Response quality**: Responses are complete and accurate

---

## 🔍 Example: Complete Workflow

```python
#!/usr/bin/env python3
"""Complete fine-tuning dataset preparation workflow."""

import json
import random
from pathlib import Path
from typing import List, Dict

def load_recipes(filepath: str) -> List[Dict]:
    """Load recipe data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_instructions(dish_name: str) -> List[str]:
    """Generate instruction variations."""
    templates = [
        f"How do I make {dish_name}?",
        f"Give me a recipe for {dish_name}.",
        f"Share the complete recipe for {dish_name}.",
        f"I want to cook {dish_name}. What are the steps?",
        f"Help me with the {dish_name} recipe.",
        f"Walk me through making {dish_name}.",
    ]
    return random.sample(templates, 3)

def create_response(recipe: Dict) -> str:
    """Create structured response."""
    parts = [f"**Title:** {recipe['name']}", ""]
    
    parts.append("**Ingredients:**")
    for ing in recipe['ingredients']:
        parts.append(f"- {ing}")
    parts.append("")
    
    parts.append("**Instructions:**")
    for i, step in enumerate(recipe['instructions'], 1):
        parts.append(f"{i}. {step}")
    parts.append("")
    
    if recipe.get('serving'):
        parts.append(f"**Serving Suggestion:** {recipe['serving']}")
    
    return "\n".join(parts)

def validate(item: Dict) -> bool:
    """Validate instruction-response pair."""
    if not item.get('instruction') or not item.get('response'):
        return False
    required = ["**Title:**", "**Ingredients:**", "**Instructions:**"]
    return all(r in item['response'] for r in required)

def main():
    # Load recipes
    recipes = load_recipes("recipes.json")
    print(f"Loaded {len(recipes)} recipes")
    
    # Generate dataset
    dataset = []
    for recipe in recipes:
        response = create_response(recipe)
        for instruction in generate_instructions(recipe['name']):
            dataset.append({
                "instruction": instruction,
                "response": response
            })
    
    # Validate and filter
    valid = [item for item in dataset if validate(item)]
    print(f"Valid pairs: {len(valid)}")
    
    # Shuffle for better training
    random.shuffle(valid)
    
    # Save
    with open("finetune_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("✓ Dataset saved to finetune_dataset.jsonl")
    
    # Print stats
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {len(valid)}")
    print(f"  Unique recipes: {len(recipes)}")
    print(f"  Avg variations per recipe: {len(valid) / len(recipes):.1f}")

if __name__ == "__main__":
    main()
```

---

## 🎓 Training Configuration

When fine-tuning, use these settings:

```python
FINETUNE_CONFIG = {
    # Fewer epochs than pre-training
    "num_train_epochs": 3,
    
    # Lower learning rate (10x lower than pre-training)
    "learning_rate": 1e-5,
    
    # Smaller batch size (instruction data is denser)
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    
    # Warmup helps stability
    "warmup_ratio": 0.1,
    
    # Mixed precision for efficiency
    "fp16": True,
}
```

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Why It's Bad | Solution |
|---------|--------------|----------|
| Too few variations | Poor generalization | 3-5 phrasings per concept |
| Inconsistent responses | Model outputs vary | Use strict templates |
| Robotic instructions | Doesn't match real users | Use natural language |
| Missing edge cases | Fails on unusual requests | Include challenging examples |
| High learning rate | Forgets pre-training | Use 1e-5 or lower |
| Too many epochs | Overfits to instructions | 1-3 epochs usually enough |

---

## 📈 Measuring Success

After fine-tuning, evaluate:

1. **Format compliance**: Does output follow structure?
2. **Instruction following**: Does it answer what was asked?
3. **Generalization**: Does it handle new phrasings?
4. **Quality**: Are responses coherent and accurate?

```python
def evaluate_output(instruction: str, response: str) -> dict:
    """Simple evaluation metrics."""
    
    scores = {
        "has_title": "**Title:**" in response,
        "has_ingredients": "**Ingredients:**" in response,
        "has_instructions": "**Instructions:**" in response,
        "reasonable_length": 100 < len(response) < 5000,
        "no_repetition": response.count("1.") <= 20,  # Not too many steps
    }
    
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores
```

---

## 🎓 Summary

1. **Define** instruction categories and templates
2. **Generate** diverse instruction variations
3. **Structure** responses consistently
4. **Validate** every pair meets requirements
5. **Balance** coverage across categories
6. **Train** with lower learning rate and fewer epochs

The key to good fine-tuning is diversity in instructions + consistency in responses!
