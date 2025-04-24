from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from .models import UploadedImage

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Load the trained model and class indices
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../potato_disease_mobilenet_model.h5')

CLASS_INDICES = {'Early_blight': 0, 'Late_blight': 1, 'healthy': 2}  # Ensure healthy is last for softmax

# Treatment suggestions for each disease
TREATMENTS = {
    'Early_blight': {
        'organic': [
            'Remove and destroy infected leaves',
            'Apply neem oil spray every 7-10 days',
            'Use copper-based fungicides',
            'Practice crop rotation'
        ],
        'chemical': [
            'Chlorothalonil-based fungicides',
            'Mancozeb-based fungicides',
            'Azoxystrobin-based fungicides'
        ]
    },
    'Late_blight': {
        'organic': [
            'Remove and destroy infected plants',
            'Apply copper fungicides',
            'Improve air circulation',
            'Avoid overhead watering'
        ],
        'chemical': [
            'Metalaxyl-based fungicides',
            'Famoxadone + cymoxanil fungicides',
            'Fluazinam-based fungicides'
        ]
    },
    'healthy': {
        'organic': [
            'Maintain good plant hygiene',
            'Use balanced organic fertilizers',
            'Practice crop rotation'
        ],
        'chemical': [
            'Apply preventive fungicides if needed',
            'Use balanced chemical fertilizers'
        ]
    }
}

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
            return model
        else:
            print(f"Model file not found at {MODEL_PATH}. Please ensure the model file exists at the correct location.")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_model()
if model is None:
    print("Warning: Model could not be loaded. Predictions will not work.")

def preprocess_image(image_path, target_size=(128, 128)):
    img_array = None  # Initialize img_array to None
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize to 128x128 as expected by the model
        img_array = np.array(img) / 255.0
        print(f"Image size after resizing: {img_array.shape}")

        # Ensure the image has 3 channels (RGB)
        if img_array.ndim == 2:  # Grayscale image
            img_array = np.stack((img_array,)*3, axis=-1)

        # Reshape to include batch dimension
        img_array = img_array.reshape((1, target_size[0], target_size[1], 3))  # Correctly reshape to (1, height, width, channels)

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None  # Return None to indicate failure

def predict_image(image_path):
    if model is None:
        print("ERROR: Model is None")
        return "Model not found"
    
    try:
        print(f"Processing image: {image_path}")
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return "Error processing image"
        
        print(f"Processed image shape: {processed_image.shape}")
        
        # Verify image content
        if np.max(processed_image) == 0:
            print("Warning: Image appears to be blank or invalid")
            return "Invalid image content"
            
        # Additional validation for healthy images
        if np.mean(processed_image) > 0.9:  # Check if image is mostly white
            print("Warning: Image appears to be mostly white")
            return "Invalid image content - too bright"
            
        if np.mean(processed_image) < 0.1:  # Check if image is mostly black
            print("Warning: Image appears to be mostly dark")
            return "Invalid image content - too dark"

        predictions = model.predict(processed_image, verbose=0)

        print(f"Predictions shape: {predictions.shape}")

        print(f"Raw predictions: {predictions}")
        
        # Get all class probabilities
        class_probs = {cls: float(prob) for cls, prob in zip(CLASS_INDICES.keys(), predictions[0])}
        print(f"Class probabilities: {class_probs}")
        
        predicted_class = list(CLASS_INDICES.keys())[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Additional validation for healthy class
        if predicted_class == 'healthy':
            print("Processing healthy image prediction")
            if confidence < 0.7 and all(prob < 0.5 for cls, prob in class_probs.items() if cls != 'healthy'):
                print("Adjusting healthy prediction confidence")
                predicted_class = 'healthy'
                confidence = class_probs['healthy']
            elif confidence < 0.5:
                print("Very low confidence in healthy prediction")
                return "Unable to determine with confidence"
        
        print(f"Prediction successful: {predicted_class} ({confidence:.2%})")
        return f"{predicted_class} ({confidence:.2%})"
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return f"Error processing image: {str(e)}"

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('upload_image')

    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def upload_image(request):
    if request.method == 'POST':
        try:
            if 'image' not in request.FILES:
                return JsonResponse({'status': 'error', 'message': 'No image file provided'}, status=400)
                
            image_file = request.FILES['image']
            
            # Validate image file type
            if not image_file.content_type.startswith('image/'):
                return JsonResponse({'status': 'error', 'message': 'Invalid file type. Only image files are allowed.'}, status=400)
            
            # Validate image size (max 5MB)
            if image_file.size > 5 * 1024 * 1024:
                return JsonResponse({'status': 'error', 'message': 'File size too large. Maximum 5MB allowed.'}, status=400)
            
            # Validate image dimensions
            try:
                img = Image.open(image_file)
                width, height = img.size
                if width < 100 or height < 100:
                    return JsonResponse({'status': 'error', 'message': 'Image dimensions too small. Minimum 100x100 pixels required.'}, status=400)
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Invalid image file. Could not read image dimensions.'}, status=400)

            file_path = default_storage.save(os.path.join('uploads', image_file.name), image_file)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # Make prediction using full path
            prediction = predict_image(full_path)

            # Save to database
            uploaded_image = UploadedImage.objects.create(image=file_path)
            print(f"Image saved at: {full_path}")

            # Split prediction and confidence
            prediction_parts = prediction.split(' (')
            prediction_text = prediction_parts[0]
            confidence_level = prediction_parts[1].replace(')', '') if len(prediction_parts) > 1 else '0%'
            
            # Render results template with prediction data
            context = {
                'status': 'success',
                'prediction': prediction_text,
                'confidence': confidence_level,
                'image_url': uploaded_image.image.url,
                'treatments': TREATMENTS.get(prediction_text, {
                    'organic': ['No specific treatment needed'],
                    'chemical': ['No specific treatment needed']
                })
            }
            print(f"Prediction context: {context}")  # Debug logging
            return render(request, 'detection/results.html', context, content_type='text/html')

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    print("Rendering upload form")
    return render(request, 'detection/upload.html', content_type='text/html')
