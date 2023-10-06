import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import requests
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('dhanvantari.h5')

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png','JPG','JPEG','PNG'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_plant_leaf(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        return False  # Image couldn't be loaded
    
    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a green color range in HSV
    lower_green = np.array([35, 50, 50])  # Lower bound for green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound for green in HSV
    
    # Create a mask to extract green regions
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # Calculate the percentage of green pixels in the image
    green_pixel_percentage = (np.count_nonzero(mask) / mask.size) * 100
    
    # Set a threshold for green pixel percentage (adjust as needed)
    green_threshold = 2  # For example, require at least 5% of green pixels
    
    # Find contours in the green mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if the image has enough green pixels to resemble a plant leaf
    # and if it contains at least one contour (leaf-like structure)
    return green_pixel_percentage >= green_threshold and len(contours) > 0

# Function to generate three outputs based on the input image
def generate_outputs(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Use your trained model for prediction here
    prediction = model.predict(img)
    disease_class = np.argmax(prediction)

    result = open("C:\\Users\\saibalaji\\Desktop\\alpha-zero\\labels.txt", 'r').readlines()
    treatment = open("C:\\Users\\saibalaji\\Desktop\\alpha-zero\\solutions.txt", 'r').readlines()
    class_name = result[disease_class]
    sol = treatment[disease_class]
    confidence_score = prediction[0][disease_class] * 100.0  # Convert to percentage
    formatted_confidence = "{:.2f}".format(confidence_score)

    # Replace these placeholders with your three output values
    print(class_name)
    print(sol)
    print(formatted_confidence)

    return class_name[2:], sol[2:], formatted_confidence

@app.route('/upload', methods=['POST'])
def upload_detect():
    if 'file' in request.files:
        # Handle file upload as usual
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            img_path = os.path.join('uploads', filename)

            if is_valid_plant_leaf(img_path):
            # Call a function to generate the three outputs (output1, output2, output3)
                output1, output2, output3 = generate_outputs(img_path)

                response_data = {
                    "output1": output1,
                    "output2": output2,
                    "output3": output3
                }

                return jsonify(response_data)
            else:
                return jsonify({"error": "Invalid image"})
        else:
            return jsonify({"error": "File format not allowed"})
    elif 'url' in request.form:
        # Handle image URL input
        img_url = request.form['url']

        # Use the requests library to fetch the image from the URL
        response = requests.get(img_url)

        if response.status_code != 200:
            return jsonify({"error": "Failed to download the image"})

        # Save the downloaded image to the 'uploads' directory
        filename = secure_filename(img_url.split("/")[-1])
        img_path = os.path.join('uploads', filename)
        with open(img_path, 'wb') as img_file:
            img_file.write(response.content)

        if is_valid_plant_leaf(img_path):
            # Call a function to generate the three outputs (output1, output2, output3)
            output1, output2, output3 = generate_outputs(img_path)

            response_data = {
                "output1": output1,
                "output2": output2,
                "output3": output3
            }

            return jsonify(response_data)
        else:
            print('invaild')
            return jsonify({"error": "Invalid image"})
    else:
        return jsonify({"error": "No file or URL provided"})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='10.9.2.221', port=5000)
