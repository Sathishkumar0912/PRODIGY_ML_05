
# Food Image Classification and Calorie Estimation

This project utilizes the pre-trained InceptionV3 model from TensorFlow to classify food items in images and estimate their calorie content. The predictions are based on the InceptionV3 model trained on the ImageNet dataset, and a simple dictionary is used to map the predicted food item to its corresponding calorie value.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (optional, currently not used in the code)
- NumPy

You can install the required libraries by running the following:

```
pip install tensorflow opencv-python numpy
```

## Project Structure

- `food_images/`: Folder containing subfolders with food images (e.g., `apple_pie/`, `baby_back_ribs/`).
- `food_calories.py`: Python script for predicting food items and estimating calories.
- `README.md`: This file.

## How It Works

### 1. **Food Image Classification**
The InceptionV3 model is pre-trained on the ImageNet dataset. The script takes an image, processes it to match the input format expected by the model, and predicts the food item in the image.

### 2. **Calorie Estimation**
A dictionary `food_calories` maps food items (predicted by the model) to their corresponding calorie values. You can expand this dictionary with more food items as needed.

### 3. **Folder Processing**
The script processes all subfolders in the `food_images/` directory. Each subfolder represents a specific food item, and the script will predict the food item and estimate the calories for the first image in each folder.


## Adding More Food Items

You can expand the `food_calories` dictionary with more food items and their corresponding calorie values. Just follow the format:

```
'new_food_item': calorie_value,
```
