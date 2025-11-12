import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import requests

st.set_page_config(page_title="AI Food Recipe Generator", page_icon="🍽️", layout="wide")

# -- Sidebar
st.sidebar.title("🍴 AI Food Recipe Generator")
st.sidebar.markdown("Detect ingredients from your photos and get recipes instantly! 📸")
st.sidebar.info("Start by uploading an image, scanning with webcam, or typing below.")

# Ingredient list (session)
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []

# -- TFLite Model Prediction Routine
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
        class_names = [line.strip() for line in f.readlines()]
    predicted_index = int(np.argmax(predictions[0]))
    ingredient = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])
    return ingredient, confidence, predictions[0], class_names

# -- Main App Layout
st.header("🧑‍🍳 Food Photo Ingredient Detector & Recipe Finder")

st.markdown(
    """
    **How does it work?**
    1. Upload food images OR scan with webcam. The AI detects ingredients and remembers them.
    2. Add ingredients manually in the box below if needed.
    3. Your ingredient list grows as you add more items.
    4. Click "Suggest Recipes!" to find what you can make!
    """
)

uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
camera_photo = st.camera_input("Scan ingredient with webcam")

# Manual ingredient entry
manual_item = st.text_input("Or, manually add an ingredient (type and press enter):")
if manual_item:
    manual_item_clean = manual_item.strip().capitalize()
    if manual_item_clean and manual_item_clean not in st.session_state.ingredients:
        st.session_state.ingredients.append(manual_item_clean)
        st.success(f"Added manually: **{manual_item_clean}**")
        st.balloons()

# Image classification
image_to_process = None
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    st.image(image_to_process, caption="Your uploaded food image", use_container_width=True)
elif camera_photo is not None:
    image_to_process = Image.open(camera_photo).convert("RGB")
    st.image(image_to_process, caption="Snapshot from webcam!", use_container_width=True)

if image_to_process is not None:
    ingredient, confidence, probs, class_names = predict_ingredient(image_to_process)
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

# Ingredient history & clear
st.subheader("📝 Your Ingredient List")
if st.session_state.ingredients:
    st.write(", ".join(st.session_state.ingredients))
    if st.button("Clear Ingredient List"):
        st.session_state.ingredients = []
else:
    st.info("No ingredients detected yet. Start by uploading, scanning, or typing!")

# Spoonacular API (unchanged)
api_key = "ed412236f67046e69561695dcda1173e"
if st.session_state.ingredients:
    if st.button("Suggest Recipes!"):
        recipes = get_recipes(st.session_state.ingredients, api_key)
        if recipes:
            st.subheader("🔎 Recipes Found")
            for r in recipes:
                st.markdown(f"**{r['title']}**")
                st.image(r['image'], width=200)
                st.markdown(f"[View Recipe](https://spoonacular.com/recipes/{r['id']})")
                if r.get("usedIngredients"):
                    st.write(f"Used ingredients: {', '.join([i['name'] for i in r['usedIngredients']])}")
                st.write("---")
        else:
            st.warning("No recipes found with these ingredients or API limit reached.")
