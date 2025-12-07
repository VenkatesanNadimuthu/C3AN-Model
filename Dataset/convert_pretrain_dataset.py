#!/usr/bin/env python3
"""
Convert narrative recipe text into structured format for pre-training.

This script transforms unstructured recipe text into a consistent format
with explicit field markers that the model can learn to distinguish.

Input format (current):
    [BOS] hmm, 'Recipe Name' is a Cuisine Type. You can make it in about X M + Y M. 
    The key ingredients are Ing1, Ing2, Ing3. Here's how you bring it to life: 
    Step1., Step2., Step3.[EOS]

Output format (structured):
    [BOS]
    **Title:** Recipe Name

    **Cuisine:** Cuisine Type | **Diet:** Vegetarian | **Time:** X mins prep, Y mins cook

    **Ingredients:**
    - Ingredient 1
    - Ingredient 2
    - Ingredient 3

    **Instructions:**
    1. First instruction step.
    2. Second instruction step.
    3. Third instruction step.

    **Serving Suggestion:** Serve with recommended accompaniments.
    [EOS]

Author: AI Research Engineer
Date: 2025-12-06
"""

import re
import os
from pathlib import Path
from typing import Optional, Tuple, List


def extract_recipe_name(text: str) -> Optional[str]:
    """Extract recipe name from various quote patterns."""
    patterns = [
        r"['\"]([^'\"]+Recipe[^'\"]*)['\"]",  # 'Recipe Name Recipe'
        r"['\"]([^'\"]+)['\"]",  # Any quoted text
        r"making (?:the )?([^,\.]+(?:Recipe)?)",  # "making the X Recipe"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            return name
    return None


def extract_cuisine_and_diet(text: str) -> Tuple[str, str]:
    """Extract cuisine type and dietary information."""
    cuisine = "Indian"  # Default
    diet = "Vegetarian"  # Default
    
    # Cuisine patterns
    cuisine_patterns = [
        r"from ([\w\s]+?) (?:cuisine|kitchens|Recipes)",
        r"is a ([\w\s]+?) (?:Vegetarian|Non Vegeterian|Vegan)",
        r"(Mexican|Chinese|Continental|North Indian|South Indian|Kerala|Goan|Karnataka|Coorg|Coastal Karnataka)",
    ]
    for pattern in cuisine_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            cuisine = match.group(1).strip()
            cuisine = re.sub(r'\s*Recipes?\s*', '', cuisine)
            break
    
    # Diet patterns
    if re.search(r'Non Vegeterian|Non-Veg|Meat|Chicken|Fish|Shrimp|Mutton|Beef|Pork', text, re.IGNORECASE):
        diet = "Non-Vegetarian"
    elif re.search(r'Vegan', text, re.IGNORECASE):
        diet = "Vegan"
    elif re.search(r'High Protein', text, re.IGNORECASE):
        diet = "High Protein Vegetarian"
    elif re.search(r'No Onion No Garlic|Sattvic', text, re.IGNORECASE):
        diet = "Sattvic (No Onion No Garlic)"
    
    return cuisine, diet


def extract_time(text: str) -> str:
    """Extract prep and cook time."""
    # Pattern: "X M + Y M" or "X M to prep and Y M to cook"
    patterns = [
        r"(\d+)\s*M\s*\+\s*(\d+)\s*M",
        r"(\d+)\s*M\s*(?:to prep|for prep).*?(\d+)\s*M\s*(?:to cook|for cooking)",
        r"takes?\s*(?:about\s*)?(\d+)\s*M.*?(\d+)\s*M",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prep = match.group(1)
            cook = match.group(2)
            return f"{prep} mins prep, {cook} mins cook"
    
    # Single time pattern
    single_match = re.search(r'(\d+)\s*M(?:inutes?)?', text, re.IGNORECASE)
    if single_match:
        return f"{single_match.group(1)} mins total"
    
    return "Time not specified"


def extract_ingredients(text: str) -> List[str]:
    """Extract ingredients from the text."""
    ingredients = []
    
    # Common patterns for ingredient lists
    patterns = [
        r"(?:key )?ingredients(?: are)?[:\s]+([^.]+?)(?:\.|Here's how|Follow these|Then,|To make)",
        r"You'll need[:\s]+([^.]+?)(?:\.|Here's how|Follow these|Then,)",
        r"Start with[:\s]+([^.]+?)(?:\.|Here's how|Follow these|Then,)",
        r"All you need is[:\s]+([^.]+?)(?:\.|Steps:|Here's how)",
        r"Gather your ingredients[:\s]+([^.]+?)(?:\.|Follow these)",
        r"Ingredients[:\s]+([^.]+?)(?:\.|To prepare)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            ing_text = match.group(1).strip()
            # Split by comma, accounting for parentheses
            raw_ingredients = re.split(r',\s*(?![^()]*\))', ing_text)
            for ing in raw_ingredients:
                ing = ing.strip()
                # Clean up common issues
                ing = re.sub(r'^and\s+', '', ing, flags=re.IGNORECASE)
                ing = re.sub(r'\s+', ' ', ing)
                if ing and len(ing) > 1:
                    ingredients.append(ing)
            break
    
    return ingredients


def extract_instructions(text: str) -> List[str]:
    """Extract cooking instructions from the text."""
    instructions = []
    
    # Find the instructions section
    patterns = [
        r"(?:Here's how[^:]*:|Follow these steps:|Steps:|To make it,|Then,|Instructions:|To prepare,)(.*?)(?:Serve|Pro tip|\[EOS\]|$)",
        r"To begin making[^,]*,(.*?)(?:Serve|Pro tip|\[EOS\]|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            inst_text = match.group(1).strip()
            
            # Split by sentence-ending patterns (., followed by capital or "To" or "Add" etc.)
            # First, normalize the text
            inst_text = re.sub(r'\s+', ' ', inst_text)
            
            # Split by periods followed by space and capital letter or cooking verbs
            raw_steps = re.split(r'\.,\s*(?=[A-Z])|,\s*(?=To\s|Add\s|Heat\s|Mix\s|Stir\s|Cook\s|Remove\s|Place\s|Preheat\s|Pour\s|Cover\s|Once\s|Next\s|Now\s|Finally\s|When\s|After\s|In\s)', inst_text)
            
            for i, step in enumerate(raw_steps):
                step = step.strip()
                # Clean up
                step = re.sub(r'^,\s*', '', step)
                step = re.sub(r'\.,?\s*$', '.', step)
                step = re.sub(r'^To begin making[^,]*,\s*', '', step, flags=re.IGNORECASE)
                
                # Ensure step ends with period
                if step and not step.endswith('.'):
                    step += '.'
                
                # Filter out empty or too short steps
                if step and len(step) > 10:
                    instructions.append(step)
            break
    
    return instructions


def extract_serving_suggestion(text: str) -> str:
    """Extract serving suggestion from the text."""
    patterns = [
        r"(Serve[^.]*?(?:for (?:lunch|dinner|breakfast|snack|dessert|meal)[^.]*)?\.)",
        r"(Serve[^.]+\.)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            suggestion = match.group(1).strip()
            # Clean up empty placeholders
            suggestion = re.sub(r',\s*and\s*,', ',', suggestion)
            suggestion = re.sub(r'along with\s*,\s*and\s*,', 'along with accompaniments', suggestion)
            suggestion = re.sub(r'along with\s*,\s*,', 'along with accompaniments', suggestion)
            suggestion = re.sub(r',\s*,', ',', suggestion)
            return suggestion
    
    return "Serve hot and enjoy!"


def convert_recipe(raw_text: str) -> str:
    """Convert a single recipe from narrative to structured format."""
    # Remove [BOS] and [EOS] for processing
    text = raw_text.strip()
    text = re.sub(r'^\[BOS\]\s*', '', text)
    text = re.sub(r'\s*\[EOS\]\s*$', '', text)
    
    # Extract components
    recipe_name = extract_recipe_name(text) or "Unknown Recipe"
    cuisine, diet = extract_cuisine_and_diet(text)
    time_info = extract_time(text)
    ingredients = extract_ingredients(text)
    instructions = extract_instructions(text)
    serving = extract_serving_suggestion(text)
    
    # Build structured output
    structured = "[BOS]\n"
    structured += f"**Title:** {recipe_name}\n\n"
    structured += f"**Cuisine:** {cuisine} | **Diet:** {diet} | **Time:** {time_info}\n\n"
    
    structured += "**Ingredients:**\n"
    if ingredients:
        for ing in ingredients:
            structured += f"- {ing}\n"
    else:
        structured += "- Ingredients not specified\n"
    
    structured += "\n**Instructions:**\n"
    if instructions:
        for i, step in enumerate(instructions, 1):
            structured += f"{i}. {step}\n"
    else:
        structured += "1. Instructions not available.\n"
    
    structured += f"\n**Serving Suggestion:** {serving}\n"
    structured += "[EOS]"
    
    return structured


def process_dataset(input_path: str, output_path: str) -> Tuple[int, int]:
    """Process the entire dataset and save structured output."""
    success_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        recipes = [line.strip() for line in f if line.strip()]
    
    structured_recipes = []
    
    for i, recipe in enumerate(recipes):
        try:
            structured = convert_recipe(recipe)
            structured_recipes.append(structured)
            success_count += 1
        except Exception as e:
            print(f"Error processing recipe {i+1}: {e}")
            error_count += 1
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for recipe in structured_recipes:
            # Write each recipe on a single line (for tokenizer training)
            # Replace newlines with special marker for training, then restore for readability
            single_line = recipe.replace('\n', ' \\n ')
            f.write(single_line + '\n')
    
    return success_count, error_count


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    input_file = script_dir / "processed_recipes.txt"
    output_file = script_dir / "structured_recipes_pretrain.txt"
    
    print("=" * 60)
    print("Recipe Pre-training Dataset Converter")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("-" * 60)
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    success, errors = process_dataset(str(input_file), str(output_file))
    
    print("-" * 60)
    print(f"✓ Successfully converted: {success:,} recipes")
    print(f"✗ Errors: {errors}")
    print(f"Output saved to: {output_file}")
    print("=" * 60)
    
    # Show sample output
    print("\n📄 SAMPLE STRUCTURED RECIPE:")
    print("-" * 60)
    with open(output_file, 'r', encoding='utf-8') as f:
        sample = f.readline().replace(' \\n ', '\n')
        print(sample)


if __name__ == "__main__":
    main()
