import os 
import json 
from PIL import Image 

import numpy as np  
import tensorflow as tf 
import streamlit as st  

working_dir =os.path.dirname(os.path.abspath(__file__)) 
model_path = '../plant_disease_prediction_model.h5' 

model =tf.keras.models.load_model(model_path) 

#load the class names 
class_indices =json.load(open(f"{working_dir}/class_indices.json")) 


#Function to load and preprocess the image using Pillow 

def load_and_preprocess_image(image_path,target_size=(224,224)): 
    img =Image.open(image_path) 
    img =img.resize(target_size) 
    img_array =np.array(img) 
    img_array =np.expand_dims(img_array,axis=0) 
    img_array = img_array.astype('float32') / 255 
    return img_array  


def predict_image_class(model,image_path,class_indices): 
    preprocessed_img =load_and_preprocess_image(image_path) 
    predictions =model.predict(preprocessed_img) 
    predicted_class_index =np.argmax(predictions,axis=1)[0] 
    predicted_class_name = class_indices[str(predicted_class_index) ] 
    return predict_image_class



#Streamlit App 

upload_image =st.file_uploader("Upload an Image ... ",type=['jpg','jpeg','png']) 


if upload_image is not None : 
    image =Image.open(upload_image) 
    col1,col2 =st.columns(2) 
    
    
    with col1 : 
        resised_img =image.resize((150,150)) 
        st.image(resised_img) 
    with col2 : 
        if st.button('Classify'):
            #preprocess the uploaded image and predict the class 
            prediction =predict_image_class(model,upload_image,class_indices) 
            st.success(f"prediction : {str(prediction)}") 
 