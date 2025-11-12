import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import re

st.set_page_config(page_title="Food Ingredient Detector & Recipe Suggester", page_icon="🍳", layout="centered")

# -- Minimal custom styling (theme-friendly) --
st.markdown("""
    <style>
    .ingredient-pill {
        display: inline-block;
        background: var(--secondary-background-color, #f2f3f5);
        color: var(--text-color, #222);
        font-weight: 500;
        padding: 7px 18px;
        margin: 5px 5px 5px 0;
        border-radius: 14px;
        font-size: 1.05em;
        border: 1px solid var(--block-border-color, #e0e3e6);
        letter-spacing: 0.02em;
        box-shadow: none;
    }
    .stButton>button {
        background: var(--secondary-background-color, #e8eaed);
        color: var(--text-color, #222);
        border-radius: 6px;
        border: 1px solid var(--block-border-color, #dadce0);
        padding: 9px 22px;
        font-size: 1em;
        font-weight: 600;
        box-shadow: none;
        transition: background .2s;
    }
    .stButton>button:hover {
        background: var(--primary-color-light, #d2e3fc);
        color: var(--primary-color, #1967d2);
        border: 1px solid var(--block-border-color, #b6b6b6);
    }
    .stTable {
        border-radius: 8px;
        border: 1px solid var(--block-border-color, #e0e3e6);
    }
    .stTabs [role='tab'] {
        background: var(--secondary-background-color, #f4f4f4);
        color: var(--text-color, #222);
        padding: 10px;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
        border: none;
        border-bottom: 2px solid var(--block-border-color, #e0e3e6);
    }
    .stTabs [aria-selected='true'] {
        background: var(--primary-color-light, #fff);
        color: var(--primary-color, #1967d2);
        border-bottom: 2px solid var(--primary-color, #1967d2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Food Ingredient Detector")
st.sidebar.markdown("Upload 📁, scan 📷, or type ⌨️ ingredients!")
st.sidebar.info("Cook creatively & reduce waste.")

if "ingredients" not in st.session_state:
    st.session_state.ingredients = []

def predict_ingredient(image_data):
    img = image_data.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1
    input_data = np.expand_dims(img_array, axis=0)
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    with open("labels.txt", "r") as f:
        class_names = [re.sub(r"^\s*\d+\s*", "", line.strip()) for line in f.readlines()]
    predicted_index = int(np.argmax(predictions[0]))
    ingredient = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])
    return ingredient, confidence, predictions[0], class_names

def get_recipes(ingredient_list, api_key):
    ingredients_str = ",".join(ingredient_list)
    url = ("https://api.spoonacular.com/recipes/findByIngredients"
           f"?ingredients={ingredients_str}&number=5&apiKey={api_key}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

st.markdown("<h1 style='text-align:center;'>Food Ingredient Detector & Recipe Suggester</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; font-size:1.07em; margin-bottom:18px;'>
    AI detects food from uploaded images, webcam photos, or manual entries.<br>
    Instantly build your ingredient list and discover new recipes.
    </div>
    """, unsafe_allow_html=True
)

# --- Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Webcam Scan", "⌨️ Manual Entry"])

with tab1:
    uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", use_column_width=False, width=180)
        ingredient, confidence, probs, class_names = predict_ingredient(image)
        if confidence > 0.4 and ingredient not in st.session_state.ingredients:
            st.session_state.ingredients.append(ingredient)
            st.success(f"Added: **{ingredient}** ({confidence * 100:.1f}%)")
            st.balloons()
        elif ingredient in st.session_state.ingredients:
            st.info(f"{ingredient} is already in your ingredient list.")
        else:
            st.warning(f"Low confidence ({confidence * 100:.1f}%). Try a clearer photo.")
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
        top_indices = np.argsort(probs)[::-1][:3]
        top_data = {"Class": [class_names[i] for i in top_indices],
                    "Confidence (%)": [f"{probs[i] * 100:.2f}" for i in top_indices]}
        st.markdown("##### Top AI Predictions")
        st.table(pd.DataFrame(top_data))

with tab2:
    with st.expander("Activate webcam photo"):
        camera_photo = st.camera_input("Take photo")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        st.image(image, caption="Webcam snapshot", use_column_width=False, width=180)
        ingredient, confidence, probs, class_names = predict_ingredient(image)
        if confidence > 0.4 and ingredient not in st.session_state.ingredients:
            st.session_state.ingredients.append(ingredient)
            st.success(f"Added: **{ingredient}** ({confidence * 100:.1f}%)")
            st.balloons()
        elif ingredient in st.session_state.ingredients:
            st.info(f"{ingredient} is already in your ingredient list.")
        else:
            st.warning(f"Low confidence ({confidence * 100:.1f}%). Try a clearer photo.")
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
        top_indices = np.argsort(probs)[::-1][:3]
        top_data = {"Class": [class_names[i] for i in top_indices],
                    "Confidence (%)": [f"{probs[i] * 100:.2f}" for i in top_indices]}
        st.markdown("##### Top AI Predictions")
        st.table(pd.DataFrame(top_data))

with tab3:
    manual_item = st.text_input("Add an ingredient manually (press Enter):")
    if manual_item:
        manual_item_clean = manual_item.strip().capitalize()
        if manual_item_clean and manual_item_clean not in st.session_state.ingredients:
            st.session_state.ingredients.append(manual_item_clean)
            st.success(f"Added manually: **{manual_item_clean}**")
            st.balloons()

# --- Ingredient pills
st.subheader("Your Ingredient List")
if st.session_state.ingredients:
    st.markdown('<div style="margin-bottom:15px;">' + ''.join(
        [f'<span class="ingredient-pill">{i}</span>' for i in st.session_state.ingredients]) + '</div>', unsafe_allow_html=True)
    if st.button("Clear Ingredient List"):
        st.session_state.ingredients = []
else:
    st.info("No ingredients yet — upload, scan or type ingredients above.")

# --- Recipes
api_key = "ed412236f67046e69561695dcda1173e"
if st.session_state.ingredients:
    if st.button("Suggest Recipes"):
        recipes = get_recipes(st.session_state.ingredients, api_key)
        if recipes:
            st.subheader("Recipes Found")
            cols = st.columns(2)
            for idx, r in enumerate(recipes):
                with cols[idx % 2]:
                    st.markdown(f"**{r['title']}**")
                    st.image(r['image'], width=160)
                    st.markdown(f"[View Recipe](https://spoonacular.com/recipes/{r['id']})")
                    used_ingr = ', '.join([i['name'] for i in r.get("usedIngredients", [])])
                    if used_ingr:
                        st.caption(f"Used: {used_ingr}")
                    st.write("---")
        else:
            st.error("No recipes found or API rate limit reached.")

