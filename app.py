import random
from uuid import uuid4
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
    ai_images, real_images = get_images()
    selected_ai_images = random.sample(ai_images, 8)
    selected_real_images = random.sample(real_images, 8)
    images = selected_ai_images + selected_real_images
    random.shuffle(images)
    return render_template('guess.html', images=images)

@app.route('/submit', methods=['POST'])
def submit():
    selections = {}
    responses = []
    userID = str(uuid4())
    for img in request.form:
        selection = request.form[img]
        selections[img] = selection
        image_type = 1 if 'real' in img else 0  # 1 for real, 0 for AI
        answer = 1 if selection == 'real' else 0  # 1 for real, 0 for AI
        correct = 1 if answer == image_type else 0
        responses.append({
            'ID': userID,  
            'Image Name': img,
            'AI/Real': image_type,
            'Answer': answer,
            'Correct': correct
        })
    
    df = pd.DataFrame(responses)

    # Define the file path
    file_path = '/home/avjadhav/server/Zoo_server/responses.csv'
    # file_path = 'responses.csv'

    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(file_path, index=False)

    return render_template('responseguess.html', responses=responses, selections=selections)


def get_images():
    static_dir = 'static'
    ai_images = [img for img in os.listdir(static_dir) if img.startswith('ai_images')]
    real_images = [img for img in os.listdir(static_dir) if img.startswith('real_images')]
    return ai_images, real_images


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000)



# [Unit]
# Description=Gunicorn instance to serve flask app
# After=network.target

# [Service]
# User=ubuntu
# Group=www-data
# WorkingDirectory=/home/avjadhav/server/Zoo_server
# ExecStart=/home/avjadhav/server/Zoo_server/venv/bin/gunicorn -b localhost:8000 app:app
# Restart=always

# [Install]
# WantedBy=multi-user.target

# upstream Zoo_server {
#     server 127.0.0.1:8000;
# }
