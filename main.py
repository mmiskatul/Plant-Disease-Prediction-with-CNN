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


