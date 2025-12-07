# Recipe Dataset Format Specification

This document defines the **industry-standard structured format** for recipe datasets used in GPT-2 pre-training and instruction fine-tuning.

---

## Problem Statement

The original datasets had **unstructured, narrative-style text** that caused the model to generate recipes where:
- Recipe Name, Ingredients, and Instructions were mixed up
- No clear boundaries between sections
- Inconsistent formatting made it difficult for the model to learn distinct patterns

---

## Solution: Structured Format with Explicit Field Markers

Using **explicit field markers** (`**Title:**`, `**Ingredients:**`, `**Instructions:**`) teaches the model:
1. Clear section boundaries
2. Consistent output structure
3. Bulleted ingredients and numbered steps

---

## Pre-training Dataset Format

### File: `structured_recipes_pretrain.txt`

Each recipe is on a **single line** with `\n` markers for internal newlines:

```
[BOS] \n **Title:** Recipe Name \n \n **Cuisine:** ... | **Diet:** ... | **Time:** ... \n \n **Ingredients:** \n - Ingredient 1 \n - Ingredient 2 \n \n **Instructions:** \n 1. Step one. \n 2. Step two. \n \n **Serving Suggestion:** Serve hot. \n [EOS]
```

### Readable Format (for reference):

```
[BOS]
**Title:** Roasted Peppers And Mushroom Tortilla Pizza Recipe

**Cuisine:** Mexican | **Diet:** Vegetarian | **Time:** 15 mins prep, 15 mins cook

**Ingredients:**
- Tortillas
- Extra Virgin Olive Oil
- Garlic
- Mozzarella cheese
- Red Yellow or Green Bell Pepper (Capsicum)
- Onions
- Kalmatta olives
- Button mushrooms

**Instructions:**
1. Turn your oven setting to broiler and keep it ready.
2. Grease your cast iron skillet with some olive oil and heat it over high heat.
3. When it starts to shimmer, bring down the heat to medium.
4. Add olive oil and minced garlic to the marinara sauce and stir to combine, keep aside.
5. Place one tortilla on the cast iron skillet.
6. Spread a thin layer of marinara sauce over the tortilla.
7. Spread the cheese as needed.
8. When the cheese starts to melt, add in the toppings (onion, mushrooms, olives, bell peppers).
9. Shift the cast iron skillet to the oven which is already under broiler setting.
10. When you see the cheese all melted and top starting to brown, remove and slice to serve.

**Serving Suggestion:** Serve Roasted Peppers And Mushroom Tortilla Pizza along with a fresh salad for a weekend night dinner.
[EOS]
```

### Key Formatting Rules:

| Element | Format | Example |
|---------|--------|---------|
| Recipe Start | `[BOS]` | `[BOS]` |
| Recipe End | `[EOS]` | `[EOS]` |
| Title | `**Title:** Name` | `**Title:** Butter Chicken` |
| Metadata | `**Cuisine:** X \| **Diet:** Y \| **Time:** Z` | `**Cuisine:** North Indian \| **Diet:** Non-Vegetarian \| **Time:** 20 mins prep, 30 mins cook` |
| Ingredients | Bulleted list with `-` | `- Chicken\n- Butter\n- Tomatoes` |
| Instructions | Numbered list | `1. Step one.\n2. Step two.` |
| Serving | Single line | `**Serving Suggestion:** Serve hot with naan.` |

---

## Fine-tuning Dataset Format (Alpaca-style JSONL)

### File: `structured_recipes_finetune.jsonl`

Each line is a JSON object with `instruction` and `response` fields:

```json
{"instruction": "Help me with the 'Butter Chicken' recipe.", "response": "**Title:** Butter Chicken\n\n**Ingredients:**\n- Chicken (500g)\n- Butter (4 tbsp)\n- Tomato puree (1 cup)\n- Heavy cream (1/2 cup)\n- Garam masala (1 tsp)\n- Kashmiri red chili powder (1 tsp)\n- Salt to taste\n\n**Instructions:**\n1. Marinate chicken with yogurt and spices for 2 hours.\n2. Cook marinated chicken in a pan until golden.\n3. In another pan, melt butter and add tomato puree.\n4. Cook until oil separates, about 10 minutes.\n5. Add cream and spices, simmer for 5 minutes.\n6. Add cooked chicken and mix well.\n7. Cook for another 10 minutes until gravy thickens.\n\n**Serving Suggestion:** Serve hot with butter naan or steamed rice."}
```

### Instruction Variations:

The dataset includes diverse instruction patterns:

| Pattern | Example |
|---------|---------|
| Direct request | `"How do I make Butter Chicken?"` |
| Help format | `"Help me with the 'Butter Chicken' recipe."` |
| Ingredient focus | `"What ingredients do I need for Butter Chicken?"` |
| Step-by-step | `"Walk me through making Butter Chicken."` |
| Complete recipe | `"Share the complete recipe for Butter Chicken."` |

---

## Inference Prompt Template

When using the fine-tuned model, format prompts as:

```
### Instruction:
{user_request}

### Response:

```

The model will generate structured output starting with `**Title:**`.

---

## Comparison: Before vs After

### BEFORE (Unstructured - causes mixed-up output):

```
[BOS] hmm, 'Butter Chicken' is a North Indian Non-Vegetarian favorite. 
You can make it in about 20 M + 30 M. The key ingredients are Chicken, 
Butter, Tomato puree, Heavy cream, Garam masala, Salt. Here's how you 
bring it to life: To begin making Butter Chicken, marinate chicken 
with yogurt and spices for 2 hours., Cook marinated chicken in a pan 
until golden., In another pan, melt butter and add tomato puree., Cook 
until oil separates., Add cream and spices., Add cooked chicken and 
mix well., Serve Butter Chicken with naan or rice.[EOS]
```

**Problems:**
- Recipe name embedded in narrative
- Ingredients as comma-separated text
- Instructions as run-on sentences
- No clear section boundaries

### AFTER (Structured - clean output):

```
[BOS]
**Title:** Butter Chicken

**Cuisine:** North Indian | **Diet:** Non-Vegetarian | **Time:** 20 mins prep, 30 mins cook

**Ingredients:**
- Chicken (500g)
- Butter (4 tbsp)
- Tomato puree (1 cup)
- Heavy cream (1/2 cup)
- Garam masala (1 tsp)
- Salt to taste

**Instructions:**
1. Marinate chicken with yogurt and spices for 2 hours.
2. Cook marinated chicken in a pan until golden.
3. In another pan, melt butter and add tomato puree.
4. Cook until oil separates, about 10 minutes.
5. Add cream and spices, simmer for 5 minutes.
6. Add cooked chicken and mix well.
7. Cook for another 10 minutes until gravy thickens.

**Serving Suggestion:** Serve hot with butter naan or steamed rice.
[EOS]
```

**Benefits:**
- ✅ Clear section markers
- ✅ Bulleted ingredients
- ✅ Numbered instructions
- ✅ Consistent structure
- ✅ Model learns distinct patterns for each section

---

## Dataset Statistics

| Dataset | Original Count | Converted Count | Format |
|---------|---------------|-----------------|--------|
| Pre-training | 8,677 recipes | 8,677 recipes | `structured_recipes_pretrain.txt` |
| Fine-tuning | 24,132 pairs | 24,132 pairs | `structured_recipes_finetune.jsonl` |

---

## Re-training Workflow

1. **Pre-train** on `structured_recipes_pretrain.txt` (7 epochs, vocab_size=12,000)
2. **Fine-tune** on `structured_recipes_finetune.jsonl` (3 epochs with lower LR)
3. **Inference** will produce structured output with clear Recipe Name, Ingredients, and Instructions

---

## Files Generated

```
Dataset/
├── processed_recipes.txt              # Original pre-training data (unstructured)
├── processed_recipes_alpaca.jsonl     # Original fine-tuning data (unstructured)
├── structured_recipes_pretrain.txt    # ✨ NEW: Structured pre-training data
├── structured_recipes_finetune.jsonl  # ✨ NEW: Structured fine-tuning data
├── convert_pretrain_dataset.py        # Converter script for pre-training
├── convert_finetune_dataset.py        # Converter script for fine-tuning
└── DATA_FORMAT_SPECIFICATION.md       # This documentation
```

---

## Next Steps

1. Re-run the tokenizer training on `structured_recipes_pretrain.txt`
2. Re-train the GPT-2 model on the structured data
3. Fine-tune on `structured_recipes_finetune.jsonl`
4. Update `app.py` inference to expect structured output format
