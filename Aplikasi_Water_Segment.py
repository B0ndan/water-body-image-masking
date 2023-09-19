import streamlit as st
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import io

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Auto_Encoder.h5')
    return model

def read_image_from_uploaded_file(uploaded_file, save_path="saved_image"):
    # Create a save path based on the type of the uploaded file
    extension = os.path.splitext(uploaded_file.name)[-1]  # Extract the file extension
    full_save_path = save_path + extension

    # Save the uploaded image
    contents = uploaded_file.read()
    with open(full_save_path, "wb") as f:
        f.write(contents)

    # Load the image using PIL
    image = Image.open(io.BytesIO(contents))
    
    # Resize the image to the expected input size for the model
    resized_image = image.resize((180, 180))

    # Convert resized image to numpy array
    return np.array(resized_image)

def main():
    st.title('Satellite Image Water Body Mask Prediction')
    st.image('water_body_1.jpg', width=400)
    st.text('This app made by Muhammad Bondan Vitto Ramadhan')

    # Upload the test image
    uploaded_file = st.file_uploader("Choose a satellite image", type=["tif", "jpg", "png"])

    if uploaded_file is not None:
        test_image = read_image_from_uploaded_file(uploaded_file)
        st.image(test_image, caption="Uploaded Satellite Image", use_column_width=True, clamp=True)

        # Load the trained model and make predictions
        model = load_model()
        predicted_mask = model.predict(np.expand_dims(test_image, axis=0))[0]

        st.image(predicted_mask, caption="Predicted Mask", use_column_width=True, clamp=True)

        # Assuming that the mask outputs values between 0 and 1
        # Calculate accuracy based on a threshold, for example 0.5
        accuracy = np.mean((predicted_mask > 0.5) == (test_image > 127.5))
        st.write(f"Mask Accuracy (based on thresholding): {accuracy:.2%}")

if __name__ == "__main__":
    main()

