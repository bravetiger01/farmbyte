import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('plant_disease_detection.h5')

# Define the class names based on your model
class_names = ['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Soybean___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


def ImageProcessing(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    
    if img is None:
        print("Error: Image not found or cannot be read.")
        return
    
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required size (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize the image (rescale pixel values to [0, 1])
    img_normalized = img_resized / 255.0
    
    # Convert the image to a batch format
    input_arr = np.array([img_normalized])
    
    # Perform prediction
    prediction = model.predict(input_arr)
    
    # Get the index of the highest probability class
    result_index = np.argmax(prediction)
    
    # Retrieve the class name corresponding to the highest probability
    model_prediction = class_names[result_index]
    
    return model_prediction

img = r"test\TomatoEarlyBlight2.JPG"
ImageProcessing(img)