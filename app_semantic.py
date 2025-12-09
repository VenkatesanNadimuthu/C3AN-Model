"""
🍳 Gourmet GPT: Semantic Recipe Assistant
A Streamlit chatbot with intent understanding, entity extraction, and smart recipe matching.

Author: AI Research Engineer
Date: 2025-12-07
"""

import streamlit as st
import torch
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gourmet GPT: Semantic Recipe Assistant",
    page_icon="🍳",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    /* Pipeline step cards */
    .pipeline-step {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        margin: 0.5rem 0; border-left: 4px solid #667eea;
    }
    .pipeline-step.intent { border-left-color: #28a745; }
    .pipeline-step.entities { border-left-color: #fd7e14; }
    .pipeline-step.search { border-left-color: #17a2b8; }
    .pipeline-step.ranking { border-left-color: #6f42c1; }
    
    .step-title { font-weight: 600; color: #333; margin-bottom: 0.5rem; }
    .step-content { color: #555; }
    
    /* Tag styling */
    .tag {
        display: inline-block; padding: 0.2rem 0.6rem; margin: 0.2rem;
        border-radius: 20px; font-size: 0.85rem; font-weight: 500;
    }
    .tag-ingredient { background: #d4edda; color: #155724; }
    .tag-action { background: #fff3cd; color: #856404; }
    .tag-cuisine { background: #cce5ff; color: #004085; }
    .tag-diet { background: #f8d7da; color: #721c24; }
    .tag-dish { background: #e2d5f1; color: #4a235a; }
    
    /* Recipe card */
    .recipe-card {
        background: white; border-radius: 10px; padding: 1.5rem;
        margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e8e8e8;
    }
    .recipe-card h3 { color: #333; margin-bottom: 0.5rem; }
    .recipe-score { 
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.85rem; float: right;
    }
    
    .recipe-section {
        margin: 1rem 0; padding: 0.75rem;
        background: #f8f9fa; border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    .recipe-section h4 { color: #667eea; margin: 0 0 0.5rem 0; }
    .recipe-section ul, .recipe-section ol { margin: 0.5rem 0 0 1.5rem; padding: 0; }
    .recipe-section li { margin: 0.3rem 0; }
    
    .footer { text-align: center; padding: 1rem; color: #6c757d; font-size: 0.85rem; margin-top: 2rem; border-top: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SEMANTIC UNDERSTANDING COMPONENTS
# ============================================================================

@dataclass
class Intent:
    """Represents the understood intent from user query."""
    type: str  # 'recipe_search', 'recipe_by_name', 'ingredient_based', 'cuisine_based', 'random', 'help'
    confidence: float
    description: str


@dataclass  
class ExtractedEntities:
    """Extracted semantic entities from user query."""
    ingredients: list = field(default_factory=list)
    cooking_actions: list = field(default_factory=list)
    cuisines: list = field(default_factory=list)
    dish_types: list = field(default_factory=list)
    dietary_preferences: list = field(default_factory=list)
    recipe_name: Optional[str] = None
    time_constraint: Optional[str] = None


@dataclass
class RankedRecipe:
    """A recipe with relevance score."""
    title: str
    ingredients: list
    instructions: list
    serving_suggestion: str
    tip: str
    score: float
    match_reasons: list


# Knowledge bases for entity extraction
COMMON_INGREDIENTS = {
    # Proteins
    'chicken', 'beef', 'pork', 'lamb', 'fish', 'shrimp', 'prawns', 'egg', 'eggs', 'tofu', 'paneer',
    # Vegetables
    'tomato', 'tomatoes', 'onion', 'onions', 'garlic', 'ginger', 'potato', 'potatoes', 'carrot', 
    'carrots', 'spinach', 'cabbage', 'cauliflower', 'broccoli', 'peas', 'beans', 'mushroom', 
    'mushrooms', 'capsicum', 'bell pepper', 'eggplant', 'brinjal', 'cucumber', 'lettuce',
    'pumpkin', 'corn', 'zucchini', 'okra', 'bhindi',
    # Grains & Carbs
    'rice', 'pasta', 'noodles', 'bread', 'flour', 'wheat', 'oats', 'quinoa', 'millet',
    # Dairy
    'milk', 'cheese', 'butter', 'cream', 'yogurt', 'curd', 'ghee', 'mozzarella',
    # Fruits
    'mango', 'banana', 'apple', 'orange', 'lemon', 'lime', 'pineapple', 'coconut', 'berries',
    # Spices & Herbs
    'cumin', 'coriander', 'turmeric', 'chili', 'chilli', 'pepper', 'salt', 'cinnamon', 
    'cardamom', 'cloves', 'mustard', 'curry leaves', 'mint', 'basil', 'oregano', 'thyme',
    # Others
    'oil', 'olive oil', 'sugar', 'honey', 'tamarind', 'coconut milk', 'soy sauce',
}

COOKING_ACTIONS = {
    'fry', 'fried', 'frying', 'deep fry', 'stir fry', 'pan fry',
    'bake', 'baked', 'baking', 'roast', 'roasted', 'roasting',
    'grill', 'grilled', 'grilling', 'bbq', 'barbecue',
    'boil', 'boiled', 'boiling', 'steam', 'steamed', 'steaming',
    'saute', 'sauteed', 'sauteing', 'simmer', 'simmered',
    'blend', 'blended', 'mix', 'mixed', 'whisk', 'whisked',
    'marinate', 'marinated', 'pressure cook', 'slow cook',
    'chop', 'dice', 'slice', 'mince', 'grate', 'knead',
    'stuffed', 'stuffing', 'wrap', 'wrapped', 'roll', 'rolled',
}

CUISINES = {
    'indian', 'north indian', 'south indian', 'tamil', 'kerala', 'goan', 'punjabi', 'gujarati',
    'bengali', 'rajasthani', 'hyderabadi', 'chettinad', 'karnataka', 'andhra', 'maharashtrian',
    'chinese', 'thai', 'japanese', 'korean', 'vietnamese', 'asian',
    'italian', 'french', 'spanish', 'greek', 'mediterranean', 'european',
    'mexican', 'american', 'tex-mex', 'latin',
    'middle eastern', 'lebanese', 'turkish', 'arabic',
    'continental', 'fusion',
}

DISH_TYPES = {
    'curry', 'gravy', 'dry', 'sabzi', 'sabji',
    'rice', 'biryani', 'pulao', 'pulav', 'fried rice', 'khichdi',
    'bread', 'roti', 'paratha', 'naan', 'dosa', 'idli', 'uttapam',
    'soup', 'salad', 'appetizer', 'starter', 'snack',
    'dessert', 'sweet', 'cake', 'pudding', 'halwa', 'kheer',
    'drink', 'beverage', 'smoothie', 'juice', 'lassi', 'shake',
    'pizza', 'burger', 'sandwich', 'wrap', 'roll',
    'pasta', 'noodles', 'chowmein',
    'dal', 'sambar', 'rasam',
    'chutney', 'pickle', 'raita', 'sauce', 'dip',
    'breakfast', 'lunch', 'dinner', 'brunch',
}

DIETARY_PREFERENCES = {
    'vegetarian', 'vegan', 'non-vegetarian', 'non-veg', 'nonveg',
    'healthy', 'low calorie', 'low fat', 'low carb', 'keto', 'diabetic friendly',
    'gluten free', 'dairy free', 'nut free',
    'high protein', 'protein rich',
    'quick', 'easy', 'simple', 'fast', '15 minute', '30 minute',
    'spicy', 'mild', 'tangy', 'sweet', 'savory',
    'kid friendly', 'kids', 'children',
}


def understand_intent(query: str) -> Intent:
    """
    Analyze user query to understand their intent.
    """
    query_lower = query.lower().strip()
    
    # Check for specific recipe name request
    recipe_name_patterns = [
        r"(?:help me with|give me|make|prepare|cook)\s+(?:the\s+)?['\"]?(.+?)['\"]?\s*recipe",
        r"recipe\s+(?:for|of)\s+['\"]?(.+?)['\"]?(?:\s|$)",
        r"how\s+(?:to|do\s+i)\s+make\s+(.+?)(?:\?|$)",
    ]
    
    for pattern in recipe_name_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return Intent(
                type='recipe_by_name',
                confidence=0.9,
                description=f"Looking for a specific recipe: '{match.group(1).strip()}'"
            )
    
    # Check for ingredient-based request
    ingredient_patterns = [
        r"i have\s+(.+)",
        r"(?:using|with|use)\s+(?:these\s+)?(?:ingredients?:?\s*)?(.+)",
        r"what\s+(?:can|could)\s+i\s+(?:make|cook)\s+with\s+(.+)",
    ]
    
    for pattern in ingredient_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return Intent(
                type='ingredient_based',
                confidence=0.85,
                description=f"Finding recipes with specified ingredients"
            )
    
    # Check for cuisine-based request
    for cuisine in CUISINES:
        if cuisine in query_lower:
            return Intent(
                type='cuisine_based',
                confidence=0.8,
                description=f"Looking for {cuisine.title()} cuisine recipes"
            )
    
    # Check for dish type request
    for dish in DISH_TYPES:
        if dish in query_lower:
            return Intent(
                type='recipe_search',
                confidence=0.75,
                description=f"Searching for {dish} recipes"
            )
    
    # General recipe search
    if any(word in query_lower for word in ['recipe', 'make', 'cook', 'prepare', 'suggest', 'recommend']):
        return Intent(
            type='recipe_search',
            confidence=0.7,
            description="General recipe search"
        )
    
    # Help intent
    if any(word in query_lower for word in ['help', 'how', 'what can you', 'explain']):
        return Intent(
            type='help',
            confidence=0.9,
            description="User needs help or guidance"
        )
    
    # Default to recipe search
    return Intent(
        type='recipe_search',
        confidence=0.5,
        description="Attempting to find relevant recipes"
    )


def extract_entities(query: str) -> ExtractedEntities:
    """
    Extract ingredients, cooking actions, and other entities from query.
    """
    query_lower = query.lower()
    entities = ExtractedEntities()
    
    # Extract ingredients
    for ingredient in COMMON_INGREDIENTS:
        if re.search(rf'\b{re.escape(ingredient)}\b', query_lower):
            entities.ingredients.append(ingredient)
    
    # Extract cooking actions
    for action in COOKING_ACTIONS:
        if re.search(rf'\b{re.escape(action)}\b', query_lower):
            entities.cooking_actions.append(action)
    
    # Extract cuisines
    for cuisine in CUISINES:
        if re.search(rf'\b{re.escape(cuisine)}\b', query_lower):
            entities.cuisines.append(cuisine)
    
    # Extract dish types
    for dish in DISH_TYPES:
        if re.search(rf'\b{re.escape(dish)}\b', query_lower):
            entities.dish_types.append(dish)
    
    # Extract dietary preferences
    for pref in DIETARY_PREFERENCES:
        if re.search(rf'\b{re.escape(pref)}\b', query_lower):
            entities.dietary_preferences.append(pref)
    
    # Extract time constraints
    time_match = re.search(r'(\d+)\s*(?:min|minute|mins|minutes)', query_lower)
    if time_match:
        entities.time_constraint = f"{time_match.group(1)} minutes"
    elif 'quick' in query_lower or 'fast' in query_lower:
        entities.time_constraint = "quick"
    
    return entities


# ============================================================================
# RECIPE DATABASE LOADER
# ============================================================================

@st.cache_data
def load_recipe_database(filepath: str = "./Dataset/structured_recipes_finetune.jsonl") -> list:
    """Load recipes from JSONL file into searchable format."""
    recipes = []
    seen_titles = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    response = data.get('response', '')
                    
                    # Parse the structured response
                    recipe = parse_recipe_response(response)
                    if recipe and recipe['title'] not in seen_titles:
                        seen_titles.add(recipe['title'])
                        recipes.append(recipe)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        st.warning(f"Recipe database not found at {filepath}")
    
    return recipes


def parse_recipe_response(response: str) -> Optional[dict]:
    """Parse a structured recipe response into a dictionary."""
    recipe = {
        'title': '',
        'ingredients': [],
        'instructions': [],
        'serving_suggestion': '',
        'tip': '',
        'raw': response
    }
    
    # Extract title
    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n|$)', response)
    if title_match:
        recipe['title'] = title_match.group(1).strip()
    else:
        return None  # Skip if no title
    
    # Extract ingredients
    ingredients_match = re.search(
        r'\*\*Ingredients:\*\*\s*\n?([\s\S]*?)(?=\*\*Instructions:|\*\*Serving|\*\*Tip:|$)',
        response
    )
    if ingredients_match:
        ingredients_text = ingredients_match.group(1)
        items = re.findall(r'-\s*(.+?)(?=\s*-\s|\s*\*\*|$)', ingredients_text)
        recipe['ingredients'] = [item.strip() for item in items if item.strip()]
    
    # Extract instructions
    instructions_match = re.search(
        r'\*\*Instructions:\*\*\s*\n?([\s\S]*?)(?=\*\*Serving|\*\*Tip:|$)',
        response
    )
    if instructions_match:
        instructions_text = instructions_match.group(1)
        steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', instructions_text, re.DOTALL)
        recipe['instructions'] = [step.strip() for step in steps if step.strip()]
    
    # Extract serving suggestion
    serving_match = re.search(r'\*\*Serving Suggestion:\*\*\s*(.+?)(?:\n|$)', response)
    if serving_match:
        recipe['serving_suggestion'] = serving_match.group(1).strip()
    
    # Extract tip
    tip_match = re.search(r'\*\*Tip:\*\*\s*(.+?)(?:\n|$)', response)
    if tip_match:
        recipe['tip'] = tip_match.group(1).strip()
    
    # Create searchable text (lowercase for matching)
    recipe['searchable'] = ' '.join([
        recipe['title'].lower(),
        ' '.join(recipe['ingredients']).lower(),
        ' '.join(recipe['instructions']).lower()
    ])
    
    return recipe


# ============================================================================
# RECIPE SEARCH & RANKING
# ============================================================================

def search_and_rank_recipes(
    recipes: list,
    intent: Intent,
    entities: ExtractedEntities,
    top_k: int = 5
) -> list[RankedRecipe]:
    """
    Search recipes and rank by relevance to intent and entities.
    """
    scored_recipes = []
    
    for recipe in recipes:
        score = 0.0
        match_reasons = []
        searchable = recipe['searchable']
        title_lower = recipe['title'].lower()
        
        # Score based on ingredient matches
        ingredient_matches = 0
        for ing in entities.ingredients:
            if ing in searchable:
                ingredient_matches += 1
                score += 15  # High weight for ingredient match
        
        if ingredient_matches > 0:
            match_reasons.append(f"✓ {ingredient_matches} ingredient(s) matched")
        
        # Score based on cooking action matches
        action_matches = 0
        for action in entities.cooking_actions:
            if action in searchable:
                action_matches += 1
                score += 10
        
        if action_matches > 0:
            match_reasons.append(f"✓ Cooking method: {', '.join(entities.cooking_actions[:2])}")
        
        # Score based on cuisine matches
        for cuisine in entities.cuisines:
            if cuisine in title_lower or cuisine in searchable:
                score += 20
                match_reasons.append(f"✓ {cuisine.title()} cuisine")
        
        # Score based on dish type matches
        for dish in entities.dish_types:
            if dish in title_lower:
                score += 25  # High weight for dish type in title
                match_reasons.append(f"✓ {dish.title()} dish")
            elif dish in searchable:
                score += 10
        
        # Score based on dietary preference mentions
        for pref in entities.dietary_preferences:
            if pref in searchable:
                score += 12
                match_reasons.append(f"✓ {pref.title()}")
        
        # Bonus for title relevance
        query_words = set(entities.ingredients + entities.dish_types + entities.cuisines)
        title_words = set(title_lower.split())
        title_overlap = len(query_words & title_words)
        if title_overlap > 0:
            score += title_overlap * 8
        
        if score > 0:
            scored_recipes.append(RankedRecipe(
                title=recipe['title'],
                ingredients=recipe['ingredients'],
                instructions=recipe['instructions'],
                serving_suggestion=recipe['serving_suggestion'],
                tip=recipe['tip'],
                score=score,
                match_reasons=match_reasons[:4]  # Top 4 reasons
            ))
    
    # Sort by score descending
    scored_recipes.sort(key=lambda x: x.score, reverse=True)
    
    return scored_recipes[:top_k]


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_pipeline_step(step_type: str, title: str, content: str):
    """Render a pipeline step card."""
    st.markdown(f"""
    <div class="pipeline-step {step_type}">
        <div class="step-title">{title}</div>
        <div class="step-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_entity_tags(entities: ExtractedEntities):
    """Render extracted entities as colored tags."""
    tags_html = ""
    
    for ing in entities.ingredients:
        tags_html += f'<span class="tag tag-ingredient">🥬 {ing}</span>'
    
    for action in entities.cooking_actions:
        tags_html += f'<span class="tag tag-action">🍳 {action}</span>'
    
    for cuisine in entities.cuisines:
        tags_html += f'<span class="tag tag-cuisine">🌍 {cuisine}</span>'
    
    for dish in entities.dish_types:
        tags_html += f'<span class="tag tag-dish">🍽️ {dish}</span>'
    
    for pref in entities.dietary_preferences:
        tags_html += f'<span class="tag tag-diet">💚 {pref}</span>'
    
    if tags_html:
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.markdown("*No specific entities detected*")


def render_recipe_card(recipe: RankedRecipe, rank: int):
    """Render a recipe card with score and details."""
    with st.container():
        st.markdown(f"""
        <div class="recipe-card">
            <span class="recipe-score">#{rank} · Score: {recipe.score:.0f}</span>
            <h3>🍽️ {recipe.title}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Match reasons
        if recipe.match_reasons:
            st.caption(" | ".join(recipe.match_reasons))
        
        # Expandable details
        with st.expander("📖 View Full Recipe", expanded=(rank == 1)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📝 Ingredients:**")
                for ing in recipe.ingredients[:10]:  # Limit to 10
                    st.markdown(f"- {ing}")
                if len(recipe.ingredients) > 10:
                    st.caption(f"... and {len(recipe.ingredients) - 10} more")
            
            with col2:
                st.markdown("**👨‍🍳 Instructions:**")
                for i, step in enumerate(recipe.instructions[:6], 1):  # Limit to 6
                    st.markdown(f"{i}. {step[:150]}{'...' if len(step) > 150 else ''}")
                if len(recipe.instructions) > 6:
                    st.caption(f"... and {len(recipe.instructions) - 6} more steps")
            
            if recipe.serving_suggestion and recipe.serving_suggestion != 'serve.':
                st.info(f"🍴 **Serving Suggestion:** {recipe.serving_suggestion}")
            
            if recipe.tip and recipe.tip != 'taste and adjust salt as you go.':
                st.success(f"💡 **Tip:** {recipe.tip}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🍳 Gourmet GPT</h1>
    <p>Semantic Recipe Assistant — Understanding Your Cooking Intent</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    # Number of results
    top_k = st.slider("🔢 Number of Results", min_value=1, max_value=10, value=5)
    
    # Show pipeline details
    show_pipeline = st.checkbox("🔍 Show Processing Pipeline", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Database Info")
    
    # Load recipe database
    recipes = load_recipe_database()
    st.info(f"📚 **{len(recipes):,}** unique recipes loaded")
    
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    st.markdown("""
    1. **Intent Detection** - Understanding what you want
    2. **Entity Extraction** - Finding ingredients, cuisines, etc.
    3. **Smart Search** - Matching against recipe database
    4. **Relevance Ranking** - Scoring and sorting results
    """)
    
    # Clear chat
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Welcome message
st.markdown("""
<div style="background-color: #e8f4f8; border-left: 4px solid #667eea; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
    <strong>👋 Welcome!</strong> I understand your cooking intent and find the best matching recipes.<br>
    Try: <em>"I have chicken and tomatoes"</em> or <em>"Quick vegetarian pasta"</em> or <em>"South Indian breakfast"</em>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🍳"):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if query := st.chat_input("What would you like to cook today?"):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    
    with st.chat_message("assistant", avatar="🍳"):
        
        # =====================================================================
        # STEP 1: INTENT UNDERSTANDING
        # =====================================================================
        intent = understand_intent(query)
        
        if show_pipeline:
            st.markdown("### 🧠 Processing Pipeline")
            render_pipeline_step(
                "intent",
                f"Step 1: Intent Detection — {intent.type.replace('_', ' ').title()}",
                f"{intent.description} (Confidence: {intent.confidence:.0%})"
            )
        
        # =====================================================================
        # STEP 2: ENTITY EXTRACTION  
        # =====================================================================
        entities = extract_entities(query)
        
        if show_pipeline:
            entity_count = (
                len(entities.ingredients) + len(entities.cooking_actions) + 
                len(entities.cuisines) + len(entities.dish_types) + 
                len(entities.dietary_preferences)
            )
            render_pipeline_step(
                "entities",
                f"Step 2: Entity Extraction — {entity_count} entities found",
                ""
            )
            render_entity_tags(entities)
        
        # =====================================================================
        # STEP 3: SEARCH & MATCH
        # =====================================================================
        if show_pipeline:
            with st.spinner("🔍 Searching recipe database..."):
                ranked_recipes = search_and_rank_recipes(recipes, intent, entities, top_k=top_k)
            
            render_pipeline_step(
                "search",
                f"Step 3: Database Search — {len(ranked_recipes)} matches found",
                f"Searched {len(recipes):,} recipes for relevant matches"
            )
        else:
            ranked_recipes = search_and_rank_recipes(recipes, intent, entities, top_k=top_k)
        
        # =====================================================================
        # STEP 4: RANKING & RESULTS
        # =====================================================================
        if show_pipeline:
            render_pipeline_step(
                "ranking",
                "Step 4: Relevance Ranking",
                "Recipes scored and sorted by match quality"
            )
        
        st.markdown("---")
        st.markdown(f"### 🏆 Top {len(ranked_recipes)} Matching Recipes")
        
        if ranked_recipes:
            for i, recipe in enumerate(ranked_recipes, 1):
                render_recipe_card(recipe, i)
        else:
            st.warning("""
            😅 No matching recipes found. Try:
            - Using different ingredients
            - Specifying a cuisine (e.g., "Indian", "Italian")
            - Mentioning a dish type (e.g., "curry", "pasta", "soup")
            """)
        
        # Save response summary to chat history
        response_summary = f"Found **{len(ranked_recipes)}** recipes matching your request."
        if ranked_recipes:
            response_summary += f"\n\nTop match: **{ranked_recipes[0].title}** (Score: {ranked_recipes[0].score:.0f})"
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_summary
        })

# Footer
st.markdown("""
<div class="footer">
    <p>🍳 <strong>Gourmet GPT</strong> — Semantic Recipe Search & Ranking</p>
    <p>Understanding your cooking intent, not just keywords</p>
</div>
""", unsafe_allow_html=True)
