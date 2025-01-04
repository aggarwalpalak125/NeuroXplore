import os
import tensorflow as tf
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Load the TensorFlow model
model = tf.keras.models.load_model('model/model.h5')  # Adjust path as needed

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the main page (file upload form)
@app.route('/')
def index():
    return render_template('front.html') 
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# POST endpoint to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'mriScan' not in request.files:
        return 'No file part'
    
    file = request.files['mriScan']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the MRI scan
        image = Image.open(file_path)
        image = image.resize((224, 224))  # Resize to model input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Run the prediction
        prediction = model.predict(image)
        prediction_percentage = prediction[0][0] * 100  # Convert to percentage
        result = 'Healthy' if prediction[0][0] < 0.5 else 'Tumour'
        
        # Redirect to the result page
        return redirect(url_for('result', result=result, 
                                percentage=f"{prediction_percentage:.2f}", 
                                name=request.form['fullName'],
                                age=request.form['age'], 
                                gender=request.form['gender'], 
                                image_path=file_path))
    
    return 'File not allowed'

# Endpoint to display the result
@app.route('/result')
def result():
    # Retrieve query parameters
    result = request.args.get('result')
    percentage = request.args.get('percentage')
    name = request.args.get('name')
    age = request.args.get('age')
    gender = request.args.get('gender')
    image_path = request.args.get('image_path')

    # Debugging: Print the values to ensure they're passed correctly
    print(f"Result: {result}, Percentage: {percentage}, Name: {name}, Age: {age}, Gender: {gender}, Image Path: {image_path}")

    # Render the result page with all the data
    return render_template('result.html', 
                           result=result, 
                           percentage=percentage, 
                           name=name, 
                           age=age, 
                           gender=gender, 
                           image_path=image_path)


# Run the app
if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.environ.get('PORT', 5000))  # Use Render's port
    app.run(debug=True, host='0.0.0.0', port=port)
