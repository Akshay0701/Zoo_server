import random
from flask import Flask, render_template, send_file, jsonify, request, redirect, url_for
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_zebra_pattern')
def generate_zebra_pattern():
    size = 128  
    Du, Dv = 0.16, 0.08 
    F, k = 0.035, 0.060  

    u = np.ones((size, size))
    v = np.zeros((size, size))
    r = 20
    u[size//2-r:size//2+r, size//2-r:size//2+r] = 0.50
    v[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25

    u += 0.05 * np.random.random((size, size))
    v += 0.05 * np.random.random((size, size))

    def laplacian(Z):
        return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
                np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4*Z)

    steps = 5000
    for i in range(steps):
        Lu = laplacian(u)
        Lv = laplacian(v)
        uvv = u * v * v
        u += Du * Lu - uvv + F * (1 - u)
        v += Dv * Lv + uvv - (F + k) * v

    threshold = 0.5
    zebra_stripes = np.where(u > threshold, 1.0, 0.0)
    img = Image.fromarray((zebra_stripes * 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/generate_cheetah_pattern')
def generate_cheetah_pattern():
    size = 128  
    background_color = (210, 180, 140) 
    spot_color = (0, 0, 0) 
    num_spots = 150 
    spot_radius = 5 

    img = Image.new('RGB', (size, size), background_color)
    draw = ImageDraw.Draw(img)

    for _ in range(num_spots):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        bbox = [x - spot_radius, y - spot_radius, x + spot_radius, y + spot_radius]
        draw.ellipse(bbox, fill=spot_color)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/guess')
def guess():
    # Load real and AI images
    real_images = os.listdir('static')

    # Ensure there are at least 8 images of each type
    real_images = random.sample(real_images, 16)

    random.shuffle(real_images)

    return render_template('guess.html', images=real_images)

# Dictionary to track the number of times each image was clicked and not clicked
image_results = {}

@app.route('/submit_guess', methods=['POST'])
def submit_guess():
    ai_images = request.form.getlist('correct_answers')
    guesses = request.form.getlist('guesses')

    # Update the image_results dictionary based on the current submission
    for image in ai_images:
        if image in guesses:
            image_results[image] = image_results.get(image, {'clicked': 0, 'not_clicked': 0})
            image_results[image]['clicked'] += 1
        else:
            image_results[image] = image_results.get(image, {'clicked': 0, 'not_clicked': 0})
            image_results[image]['not_clicked'] += 1

    for image in [img for img in request.form.getlist('images') if img not in ai_images]:
        if image in guesses:
            image_results[image] = image_results.get(image, {'clicked': 0, 'not_clicked': 0})
            image_results[image]['clicked'] += 1
        else:
            image_results[image] = image_results.get(image, {'clicked': 0, 'not_clicked': 0})
            image_results[image]['not_clicked'] += 1

    # Convert the image_results dictionary to a DataFrame
    results_df = pd.DataFrame(image_results.values(), index=image_results.keys())

    # Reset the index
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Image'}, inplace=True)

    # Save the DataFrame to an Excel file
    results_df.to_excel('results.xlsx', index=False)

    return f"Results saved to results.xlsx."


if __name__ == '__main__':
    app.run(debug=True)
