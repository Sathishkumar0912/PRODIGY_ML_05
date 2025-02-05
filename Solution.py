# Import necessary libraries
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import os

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Define a dictionary to map food items to calorie content (You can expand this with actual data)
food_calories = {
    'apple_pie': 300,
    'baby_back_ribs': 350,
    'baklava': 400,
    'beef_carpaccio': 250,
    'beef_tartare': 200,
    'beet_salad': 150,
    'beignets': 250,
    'bibimbap': 500,
    'bread_pudding': 300,
    'breakfast_burrito': 350,
    # Add more foods and their corresponding calorie values
}

def predict_food(image_path):
    # Load the image and prepare it for prediction
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict the food item in the image
    preds = model.predict(img_array)
    
    # Decode the prediction into readable labels
    decoded_preds = decode_predictions(preds, top=1)[0][0]
    food_item = decoded_preds[1]  # Get the predicted food name
    
    # Map the predicted food item to its calorie content
    calories = food_calories.get(food_item.lower().replace(' ', '_'), 'Unknown')  # Use the mapped calorie if found
    
    return food_item, calories

def process_food_folders(folder_path):
    # Iterate over all subfolders (food items) and get the first image for recognition
    for folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder)
        
        # Check if it is a directory
        if os.path.isdir(folder_full_path):
            for file in os.listdir(folder_full_path):
                if file.endswith('.jpg'):
                    image_path = os.path.join(folder_full_path, file)
                    food_item, calories = predict_food(image_path)
                    print(f"Predicted food: {food_item}, Estimated calories: {calories}")
                    break  # Process only the first image in the folder

# Path to the folder containing your food image folders (replace with your actual folder path)
food_image_folder = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\TASK-5\food_images"  # <-- Change this path
process_food_folders(food_image_folder)
