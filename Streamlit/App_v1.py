import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Food Waste Tracker")
st.write("Upload a photo of your vegetable")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your food photo", use_container_width=True)

    # Preprocess for TFLite model
    img = image.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1
    input_data = np.expand_dims(img_array, axis=0)

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Load class labels
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    st.success(f"Predicted: **{predicted_class}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")
