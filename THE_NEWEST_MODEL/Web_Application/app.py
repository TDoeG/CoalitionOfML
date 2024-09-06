from flask import Flask, request, render_template, redirect, url_for
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.abspath('./THE_NEWEST_MODEL'))
from model import Net  # Assuming model.py is in the same directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directory to store uploaded images
UPLOAD_FOLDER = 'The_Newest_model/Web_Application/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model_path = 'THE_NEWEST_MODEL/Saved_Models/80.00_6cifar10_cnn.pth'  # Replace with your actual model path
model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Define transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 mean and std
])

# Define classes (CIFAR-10 in this case)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open the image and apply transformations
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        predicted_class = classes[predicted.item()]
        
        # Pass the filename (relative path) for the image URL
        image_url = f'uploads/{filename}'
        
        # Render result page with prediction and uploaded image path
        return render_template('result.html', predicted_class=predicted_class, image_url=image_url)
    
    return redirect(request.url)

if __name__ == '__main__':
    # Make sure the uploads folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
