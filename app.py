import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("🧠 Deepfake Image Detector (Lite Model)")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["fake", "real"]
THRESHOLD = 0.8

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocessing
    img = image.resize((128,128))
    img = np.array(img).astype("float32")

    # Normalize (adjust if needed)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run prediction
    interpreter.invoke()

    # Get output
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("Prediction Value:", float(pred))

    # Classification
    if pred > THRESHOLD:
        label = CLASS_NAMES[1]
        confidence = pred
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - pred

    # Output
    if label.lower() == "fake":
        st.error(f"⚠️ Fake Image\nConfidence: {confidence:.2f}")
    else:
        st.success(f"✅ Real Image\nConfidence: {confidence:.2f}")