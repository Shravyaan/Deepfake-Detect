# File: detector/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .model_logic import run_model_on_file
import os

def upload_and_predict(request):
    # This is the main view for both showing the form and handling the upload
    
    if request.method == 'POST':
        # --- This block runs when the user submits the form ---
        
        # 1. Get the file from the request
        uploaded_file = request.FILES['file']
        
        # 2. Save it to a temporary location
        # We need to use the 'media' folder for temporary storage
        fs = FileSystemStorage(location='media/')
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = fs.path(filename)
        
        # 3. Run the model on the saved file
        try:
            # THIS IS THE REAL MODEL CALL (NOT FAKE DATA)
            result = run_model_on_file(filepath) 
            
            # This is the debug line, it's safe to keep
            print(f"--- DEBUG: The result dictionary is: {result} ---")

            # 4. Clean up the uploaded file
            os.remove(filepath)
            
            # 5. Render the 'results.html' page and pass the result
            return render(request, 'detector/results.html', {'result': result})
            
        except Exception as e:
            # Handle any errors
            os.remove(filepath) # Clean up even if it fails
            return render(request, 'detector/index.html', {'error': str(e)})

    else:
        # --- This block runs for a GET request (visiting the page) ---
        # Just show the upload page
        return render(request, 'detector/index.html')

def home_view(request):
    # This function just renders the home.html page
    return render(request, 'detector/home.html')

def about_view(request):
    # This function just renders the about.html page
    return render(request, 'detector/about.html')