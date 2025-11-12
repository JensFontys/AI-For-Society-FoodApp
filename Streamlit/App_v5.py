import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import re

st.set_page_config(page_title="AI Food Recipe Generator", page_icon="🍽️", layout="wide")

# Custom app styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc !important;
    }
    div[data-testid="stSidebar"] {background-color: #FFE7CE;}
    .stTabs [role="tab"] {
        background: #fff6e0;
        color: #A85B18;
        padding: 10px;
        border-radius: 6px 6px 0 0;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: #FFD9A0;
        color: #d56f00;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar
st.sidebar.header("🍴 AI Food Recipe Generator")
st.sidebar.markdown("Detect ingredients from your photos, scan with your camera, or type by hand!")
st.sidebar.info("Get inspired to reduce food waste and cook smarter. 🌱")

# --- Store ingredients
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []

# --- Model prediction routine
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
    # Remove numbers and spaces before label names
    with open("labels.txt", "r") as f:
        class_names = [re.sub(r"^\s*\d+\s*", "", line.strip()) for line in f.readlines()]
    predicted_index = int(np.argmax(predictions[0]))
    ingredient = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])
    return ingredient, confidence, predictions[0], class_names

# --- Recipe API
def get_recipes(ingredient_list, api_key):
    ingredients_str = ",".join(ingredient_list)
    url = (
        "https://api.spoonacular.com/recipes/findByIngredients"
        f"?ingredients={ingredients_str}&number=5&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# --- Main layout
st.markdown("<h1 style='text-align: center;'>🍳 Food Ingredient Detector & Recipe Suggester</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
    Detect food items from images, webcam, or manual entry.<br>
    Build your ingredient list and get healthy recipes!
    </div>
    """, unsafe_allow_html=True
)

# --- Tabs for input methods
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Webcam Scan", "⌨️ Manual Entry"])

# 1. Upload Image
with tab1:
    uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded food image", use_column_width=True)
        ingredient, confidence, probs, class_names = predict_ingredient(image)
        if confidence > 0.4 and ingredient not in st.session_state.ingredients:
            st.session_state.ingredients.append(ingredient)
            st.success(f"🟢 Added ingredient: **{ingredient}** ({confidence * 100:.1f}%)")
            st.balloons()
        elif ingredient in st.session_state.ingredients:
            st.info(f"{ingredient} is already in your ingredient list.")
        else:
            st.warning(f"Low confidence ({confidence * 100:.1f}%). Try a clearer photo.")
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
        top_indices = np.argsort(probs)[::-1][:3]
        top_data = {"Class": [class_names[i] for i in top_indices],
                    "Confidence (%)": [f"{probs[i] * 100:.2f}" for i in top_indices]}
        st.subheader("Top 3 AI Predictions:")
        st.table(pd.DataFrame(top_data))

# 2. Webcam Scan in Expander for "dropdown" effect
with tab2:
    with st.expander("Activate webcam 🖼️"):
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            st.image(image, caption="Snapshot from webcam!", use_column_width=True)
            ingredient, confidence, probs, class_names = predict_ingredient(image)
            if confidence > 0.4 and ingredient not in st.session_state.ingredients:
                st.session_state.ingredients.append(ingredient)
                st.success(f"🟢 Added ingredient: **{ingredient}** ({confidence * 100:.1f}%)")
                st.balloons()
            elif ingredient in st.session_state.ingredients:
                st.info(f"{ingredient} is already in your ingredient list.")
            else:
                st.warning(f"Low confidence ({confidence * 100:.1f}%). Try a clearer photo.")
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
            top_indices = np.argsort(probs)[::-1][:3]
            top_data = {"Class": [class_names[i] for i in top_indices],
                        "Confidence (%)": [f"{probs[i] * 100:.2f}" for i in top_indices]}
            st.subheader("Top 3 AI Predictions:")
            st.table(pd.DataFrame(top_data))

# 3. Manual entry
with tab3:
    manual_item = st.text_input("Add an ingredient (press Enter):")
    if manual_item:
        manual_item_clean = manual_item.strip().capitalize()
        if manual_item_clean and manual_item_clean not in st.session_state.ingredients:
            st.session_state.ingredients.append(manual_item_clean)
            st.success(f"✅ Added manually: **{manual_item_clean}**")
            st.balloons()

# --- Ingredient history
st.subheader("📝 Your Ingredient List")
if st.session_state.ingredients:
    st.warning("Ingredients detected or added:")
    st.markdown(", ".join([f"**{i}**" for i in st.session_state.ingredients]))
    if st.button("🧹 Clear List"):
        st.session_state.ingredients = []
else:
    st.info("No ingredients yet. Start by uploading, scanning, or typing!")

# --- Recipe generator
api_key = "ed412236f67046e69561695dcda1173e"
if st.session_state.ingredients:
    if st.button("🍽️ Suggest Recipes!"):
        recipes = get_recipes(st.session_state.ingredients, api_key)
        if recipes:
            st.subheader("🔎 Recipes Found")
            cols = st.columns(2)
            for idx, r in enumerate(recipes):
                with cols[idx % 2]:
                    st.markdown(f"### {r['title']}")
                    st.image(r['image'], width=250)
                    st.markdown(f"[View Recipe](https://spoonacular.com/recipes/{r['id']})")
                    used_ingr = ', '.join([i['name'] for i in r.get("usedIngredients", [])])
                    if used_ingr:
                        st.caption(f"Used: {used_ingr}")
                    st.write("---")
        else:
            st.error("No recipes found with these ingredients or API limit reached.")
