#!/usr/bin/env python3
"""
Convert instruction-response JSONL dataset into structured format for fine-tuning.

This script transforms the Alpaca-style instruction dataset to use properly
structured responses that the model will learn to generate consistently.

Input format (current):
    {"instruction": "Help me with 'Recipe Name' recipe.", 
     "response": "Title: Recipe Name\n\nIngredients: Ing1, Ing2, Ing3\n\nMethod: Step1., Step2., Step3."}

Output format (structured):
    {"instruction": "Help me with 'Recipe Name' recipe.",
     "response": "**Title:** Recipe Name\n\n**Ingredients:**\n- Ing1\n- Ing2\n- Ing3\n\n**Instructions:**\n1. Step1.\n2. Step2.\n3. Step3.\n\n**Serving Suggestion:** Serve hot and enjoy!"}

Author: AI Research Engineer
Date: 2025-12-06
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_title(response: str) -> str:
    """Extract recipe title from response."""
    match = re.search(r'Title:\s*([^\n]+)', response)
    if match:
        return match.group(1).strip()
    return "Recipe"


def extract_ingredients_from_response(response: str) -> List[str]:
    """Extract ingredients from the response text."""
    ingredients = []
    
    # Find ingredients section
    match = re.search(r'Ingredients?:\s*([^\n]+(?:\n(?!Method:|Instructions:|Title:)[^\n]+)*)', response, re.IGNORECASE)
    if match:
        ing_text = match.group(1).strip()
        
        # Split by comma, accounting for parentheses
        raw_ingredients = re.split(r',\s*(?![^()]*\))', ing_text)
        
        for ing in raw_ingredients:
            ing = ing.strip()
            # Clean up
            ing = re.sub(r'^and\s+', '', ing, flags=re.IGNORECASE)
            ing = re.sub(r'\s+', ' ', ing)
            if ing and len(ing) > 1:
                ingredients.append(ing)
    
    return ingredients


def extract_instructions_from_response(response: str) -> List[str]:
    """Extract instructions from the response text."""
    instructions = []
    
    # Find method/instructions section
    match = re.search(r'(?:Method|Instructions?):\s*(.+?)(?:\[EOS\]|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        method_text = match.group(1).strip()
        
        # Clean up common patterns
        method_text = re.sub(r'Start with these ingredients:[^.]+\.?\s*Then,?\s*', '', method_text)
        method_text = re.sub(r'\s+', ' ', method_text)
        
        # Split by periods followed by comma or capital letter
        raw_steps = re.split(r'\.,\s*(?=[A-Z])|,\s*(?=To\s|Add\s|Heat\s|Mix\s|Stir\s|Cook\s|Remove\s|Place\s|Preheat\s|Pour\s|Cover\s|Once\s|Next\s|Now\s|Finally\s|When\s|After\s|In\s|Tip:|Pro tip)', method_text)
        
        for step in raw_steps:
            step = step.strip()
            # Clean up
            step = re.sub(r'^,\s*', '', step)
            step = re.sub(r'\.,?\s*$', '.', step)
            step = re.sub(r'^To begin making[^,]*,\s*', '', step, flags=re.IGNORECASE)
            
            # Skip serving suggestions (we'll handle separately)
            if re.match(r'^Serve\s', step, re.IGNORECASE):
                continue
            
            # Skip tips (we'll handle separately)
            if re.match(r'^(Pro )?tip:', step, re.IGNORECASE):
                continue
            
            # Ensure step ends with period
            if step and not step.endswith('.'):
                step += '.'
            
            # Filter out empty or too short steps
            if step and len(step) > 10:
                instructions.append(step)
    
    return instructions


def extract_serving_and_tips(response: str) -> Tuple[str, Optional[str]]:
    """Extract serving suggestion and tips from response."""
    serving = "Serve hot and enjoy!"
    tip = None
    
    # Find serving suggestion
    serve_match = re.search(r'(Serve[^.]*?(?:for (?:lunch|dinner|breakfast|snack|dessert|meal)[^.]*)?\.)', response, re.IGNORECASE)
    if serve_match:
        serving = serve_match.group(1).strip()
        # Clean up empty placeholders
        serving = re.sub(r',\s*and\s*,', ',', serving)
        serving = re.sub(r'along with\s*,\s*and\s*,', 'along with accompaniments', serving)
        serving = re.sub(r'along with\s*,\s*,', 'along with accompaniments', serving)
        serving = re.sub(r',\s*,', ',', serving)
    
    # Find tips
    tip_match = re.search(r'(?:Pro )?[Tt]ip:\s*([^.]+\.)', response)
    if tip_match:
        tip = tip_match.group(1).strip()
    
    return serving, tip


def convert_response(response: str) -> str:
    """Convert a single response from unstructured to structured format."""
    # Extract components
    title = extract_title(response)
    ingredients = extract_ingredients_from_response(response)
    instructions = extract_instructions_from_response(response)
    serving, tip = extract_serving_and_tips(response)
    
    # Build structured response
    structured = f"**Title:** {title}\n\n"
    
    structured += "**Ingredients:**\n"
    if ingredients:
        for ing in ingredients:
            structured += f"- {ing}\n"
    else:
        structured += "- See recipe for full ingredient list\n"
    
    structured += "\n**Instructions:**\n"
    if instructions:
        for i, step in enumerate(instructions, 1):
            structured += f"{i}. {step}\n"
    else:
        structured += "1. Follow the method described in the original recipe.\n"
    
    structured += f"\n**Serving Suggestion:** {serving}"
    
    if tip:
        structured += f"\n\n**Tip:** {tip}"
    
    return structured


def diversify_instructions(original_instruction: str, recipe_name: str) -> List[Dict[str, str]]:
    """Generate diverse instruction variations for the same recipe."""
    variations = []
    
    # Base instruction patterns
    patterns = [
        f"How do I make {recipe_name}?",
        f"Give me a recipe for {recipe_name}.",
        f"I want to cook {recipe_name}. What are the steps?",
        f"Can you share the recipe for {recipe_name}?",
        f"What ingredients do I need for {recipe_name} and how do I prepare it?",
        f"Walk me through making {recipe_name}.",
        f"I'd like to prepare {recipe_name}. Help me with the recipe.",
        f"Share the complete recipe for {recipe_name} with ingredients and instructions.",
    ]
    
    # Return 2-3 random variations (the original + 1-2 new ones)
    import random
    selected = random.sample(patterns, min(2, len(patterns)))
    
    return selected


def process_dataset(input_path: str, output_path: str, diversify: bool = True) -> Tuple[int, int]:
    """Process the entire JSONL dataset."""
    success_count = 0
    error_count = 0
    output_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                instruction = data.get('instruction', '')
                response = data.get('response', '')
                
                # Convert response to structured format
                structured_response = convert_response(response)
                
                # Create output entry
                output_entry = {
                    "instruction": instruction,
                    "response": structured_response
                }
                output_data.append(output_entry)
                success_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                error_count += 1
            except Exception as e:
                print(f"Line {line_num}: Processing error - {e}")
                error_count += 1
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return success_count, error_count


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    input_file = script_dir / "processed_recipes_alpaca.jsonl"
    output_file = script_dir / "structured_recipes_finetune.jsonl"
    
    print("=" * 60)
    print("Recipe Fine-tuning Dataset Converter")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("-" * 60)
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    success, errors = process_dataset(str(input_file), str(output_file))
    
    print("-" * 60)
    print(f"✓ Successfully converted: {success:,} instruction-response pairs")
    print(f"✗ Errors: {errors}")
    print(f"Output saved to: {output_file}")
    print("=" * 60)
    
    # Show sample output
    print("\n📄 SAMPLE STRUCTURED INSTRUCTION-RESPONSE:")
    print("-" * 60)
    with open(output_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
        print(f"INSTRUCTION: {sample['instruction']}\n")
        print(f"RESPONSE:\n{sample['response']}")


if __name__ == "__main__":
    main()
