import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

st.set_page_config(page_title="Ingredient to Recipe AI", page_icon="🍳", layout="wide")
st.title("🍳 Ingredient Detector & Recipe Generator")

# Ingredient filter list (expand as desired)
ingredient_classes = {
    "lemon", "banana", "carrot", "cucumber", "apple", "orange", "strawberry", "tomato", "grapefruit",
    "pear", "plum", "pineapple", "grape", "peach", "apricot", "mango", "fig", "olive", "broccoli",
    "pumpkin", "zucchini", "artichoke", "cauliflower", "lettuce", "corn", "beet", "asparagus", "parsnip",
    "potato", "yam", "onion", "shallot", "garlic", "pepper", "chili", "eggplant", "peas", "bean", "soybean",
    "spinach", "kale", "cabbage", "celery", "turnip", "radish", "hazelnut", "almond", "walnut", "cashew",
    "chestnut", "peanut", "pistachio", "pecan", "macadamia", "rice", "oat", "wheat", "flour", "barley",
    "buckwheat", "lentil", "chickpea", "millet", "quinoa", "sesame", "sunflower", "pumpkin seed", "eggs",
    "milk", "butter", "cheese", "cream", "yogurt", "honey", "sugar", "salt", "vanilla", "chocolate",
    "cocoa", "coffee", "tea", "basil", "thyme", "rosemary", "sage", "parsley", "cilantro", "mint", "dill",
    "oregano", "marjoram", "tarragon", "mushroom", "meat loaf", "beef", "chicken", "turkey", "duck",
    "salmon", "trout", "cod", "tuna", "shrimp", "crab", "lobster", "clam", "scallop", "oyster"
}
ingredient_classes = set([x.lower() for x in ingredient_classes])  # lowercase all

# Ingredient memory in session
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def classify_image(img):
    img = img.resize((224, 224))
    x = np.array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    top10 = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=10)[0]
    filtered = []
    for _, label, score in top10:
        label_simple = label.replace('_', ' ').replace('-', ' ').lower()
        for ingr in ingredient_classes:
            if ingr in label_simple or label_simple in ingr:
                filtered.append((label, score))
                break
    return filtered

# Show header and uploader
st.header("🧑‍🍳 Step 1: Build your ingredient list")
uploaded_file = st.file_uploader("Upload ingredient image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    ingredients = classify_image(image)
    if ingredients:
        top_label, top_score = ingredients[0]
        # Always add top predicted ingredient automatically
        if top_label not in st.session_state.ingredients and top_score > 0.15:
            st.session_state.ingredients.append(top_label)
            st.success(f"Added ingredient: **{top_label.capitalize()}** ({top_score*100:.2f}%)")
            st.balloons()
        elif top_label in st.session_state.ingredients:
            st.info(f"{top_label.capitalize()} is already in your ingredient list.")
        st.subheader("Prediction Details:")
        for label, score in ingredients[:5]:
            st.write(f"{label.capitalize()}: {score * 100:.2f}%")
    else:
        st.warning("No ingredient detected. Use a clear photo or expand filter list.")

# Show and manage ingredient list
st.subheader("📝 Your Ingredient List")
if st.session_state.ingredients:
    st.write(", ".join([x.capitalize() for x in st.session_state.ingredients]))
    if st.button("Clear Ingredient List"):
        st.session_state.ingredients = []
else:
    st.info("No ingredients detected yet. Upload some images!")

# Spoonacular API setup (insert your real API key!)
API_KEY = "ed412236f67046e69561695dcda1173e"

def get_recipes(ingredient_list, api_key):
    # Collate ingredient names into a comma-separated string
    ingredients_str = ",".join([i.replace('_', ' ') for i in ingredient_list])
    url = (
        "https://api.spoonacular.com/recipes/findByIngredients"
        f"?ingredients={ingredients_str}&number=5&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Recipe API error ({response.status_code}): {response.text[:200]}")
        return []

st.header("🍽️ Step 2: Generate recipes with your ingredients!")
if st.session_state.ingredients:
    if st.button("Suggest Recipes!"):
        recipes = get_recipes(st.session_state.ingredients, API_KEY)
        if recipes:
            st.success("Here are recipe ideas you can make!")
            for r in recipes:
                st.markdown(f"**{r['title']}**")
                st.image(r['image'], width=200)
                st.markdown(f"[View Recipe](https://spoonacular.com/recipes/{r['id']})")
                if r.get("usedIngredients"):
                    st.write(f"Used ingredients: {', '.join([i['name'] for i in r['usedIngredients']])}")
                st.write("---")
        else:
            st.warning("No recipes found with these ingredients or API limit reached.")
