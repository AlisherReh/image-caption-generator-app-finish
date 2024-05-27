import unittest
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from gtts import gTTS
import os
import base64
from deep_translator import GoogleTranslator
import time
from PIL import Image

# Create a temporary directory to store audio files
if not os.path.exists("temp"):
    os.mkdir("temp")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Load ResNet50 model
resnet50_model = ResNet50(weights="imagenet")
resnet50_model = Model(inputs=resnet50_model.inputs, outputs=resnet50_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Load the mappings between word and index
with open("data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)
with open("data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)

# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")

# Language selection
language = st.selectbox("Select language", ("English", "Kazakh", "Russian"))

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")

def update_key():
    st.session_state.uploader_key += 1

# Preprocess image
def preprocess_image(img):
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Encode image using ResNet50
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector

# Generate caption using the model
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(80):
        sequence = [word_to_index.get(w, 0) for w in inp_text.split()]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word.get(ypred, '')
        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Text-to-Speech function
def text_to_speech(text, lang='en', tld='com'):
    tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
    file_path = f"temp/{text[:20].replace(' ', '_')}.mp3"
    tts.save(file_path)
    return file_path

# Translator function
def translate_text(text, target_lang):
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return translated

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Encode image
        photo = encode_image(uploaded_image).reshape((1, 2048))

        # Generate caption
        generated_caption = predict_caption(photo)

        # Translate caption if necessary
        if language == "Kazakh":
            generated_caption = translate_text(generated_caption, 'kk')
        elif language == "Russian":
            generated_caption = translate_text(generated_caption, 'ru')

    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating speech..."):
        lang_code = 'en' if language == 'English' else 'ru'
        audio_file_path = text_to_speech(generated_caption, lang=lang_code, tld='com')

    # Autoplay audio using HTML
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{base64.b64encode(open(audio_file_path, 'rb').read()).decode()}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

    # Clean up the temporary audio file
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    # Wait for 2-3 seconds before rerunning
    time.sleep(4)
    
    update_key()
    st.experimental_rerun()

class TestApp(unittest.TestCase):

    @staticmethod
    def create_test_image(file_path):
        # Create a test image
        test_image = Image.new("RGB", (224, 224), color="white")
        test_image.save(file_path)

    def setUp(self):
        # Create a test image before each test
        self.img_path = "test_image.jpg"
        self.create_test_image(self.img_path)

    def tearDown(self):
        # Remove the test image after each test
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_preprocess_image(self):
        # Test preprocess_image function
        preprocessed_img = preprocess_image(self.img_path)
        # Check if the output shape matches
        self.assertEqual(preprocessed_img.shape, (1, 224, 224, 3))

    def test_encode_image(self):
        # Test encode_image function
        encoded_img = encode_image(self.img_path)
        # Check if the output shape matches
        self.assertEqual(encoded_img.shape, (1, 2048))

    def test_predict_caption(self):
        # Test predict_caption function
        photo = encode_image(self.img_path).reshape((1, 2048))
        caption = predict_caption(photo)
        # Check if the output is a string
        self.assertIsInstance(caption, str)

    def test_text_to_speech(self):
        # Test text_to_speech function
        text = "This is a test."
        audio_path = text_to_speech(text)
        # Check if the audio file exists
        self.assertTrue(os.path.exists(audio_path))
        # Remove the test audio file
        os.remove(audio_path)

    def test_translate_text(self):
        # Test translate_text function
        translated_text = translate_text("Hello", "fr")
        # Check if the output is a string
        self.assertIsInstance(translated_text, str)

if __name__ == '__main__':
    # Run the unit tests and display the results
    unittest.main()
