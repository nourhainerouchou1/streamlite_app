import streamlit as st
#import tensorflow as tf
import numpy as np
from PIL import Image


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('VStrained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = 'frigo_illu.png'
    image = Image.open(image_path)
    st.image(image, caption="Fruits & Vegetables Recognition System", use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Project")


# Displaying Images
    # Displaying Images
    st.subheader("Visualisation des données")
    fruits_images = [
        'matr.png',
        'vis.png',
        'vis2.png'
    ]

    captions = ["Confusion Matrix", "Training Accuracy & Validation Accuracy", "Training Loss & Validation Loss"]

    for img_path, caption in zip(fruits_images, captions):
        image = Image.open(img_path)
        st.image(image, caption=caption, use_column_width=True)
    # Rest of the content
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)
    
    #Predict button
    if(st.button("Predict")):
        st.snow()      
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = [i[:-1] for i in content]
        
        # Making the prediction result bold
        prediction_result = "**Model is Predicting it's a {}**".format(label[result_index])
        st.success(prediction_result)
