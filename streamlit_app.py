import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


st.title('Waste Classification')
st.header("Please choose a picture")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
uploaded_file = st.file_uploader("Please choose a picture",label_visibility="hidden")
path = "upload.jpg"

model =  tf.keras.models.load_model('waste_classification.h5', compile = False)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='filename: '+ uploaded_file.name)
    # resize = tf.image.resize(np.asarray(img), (256,256))
    img = img.save(path)
    image = tf.keras.utils.load_img(
    path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

