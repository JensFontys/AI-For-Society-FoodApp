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
st.sidebar.info("Start by uploading a food photo below.")

# -- STORE Ingredient list in session
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

# -- SPOONACULAR API Integration
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

# -- Main App Layout
st.header("🧑‍🍳 Food Photo Ingredient Detector & Recipe Finder")
st.markdown(
    """
    **How does it work?**
    1. Upload food images. The AI detects ingredients and remembers them.
    2. Your ingredient list grows as you upload more items.
    3. Click "Suggest Recipes!" to find what you can make!
    """
)

uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded food image", use_container_width=True)
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

    # Show top 3 predictions (no chart needed)
    top_indices = np.argsort(probs)[::-1][:3]
    top_data = {"Class": [class_names[i] for i in top_indices],
                "Confidence (%)": [f"{probs[i] * 100:.2f}" for i in top_indices]}
    st.subheader("Top 3 AI Predictions:")
    st.table(pd.DataFrame(top_data))

# --- Ingredient History + Remove/Clear
st.subheader("📝 Your Ingredient List")
if st.session_state.ingredients:
    st.write(", ".join(st.session_state.ingredients))
    if st.button("Clear Ingredient List"):
        st.session_state.ingredients = []
else:
    st.info("No ingredients detected yet. Start by uploading a photo!")

# --- Spoonacular Recipe API Integration
api_key = "ed412236f67046e69561695dcda1173e"  # <--- Put your real key here!
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
