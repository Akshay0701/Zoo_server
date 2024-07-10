import random
import subprocess
from uuid import uuid4
from flask import Flask, render_template, send_file, jsonify, request, redirect, url_for
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import pandas as pd
import logging

# from lammps import generate_model, write_lammps_input, run_lammps_simulation, create_image_from_lammps_output

# Paths
output_folder_path = 'outputImage'
binary_image_path = os.path.join(output_folder_path, 'binary_image.png')
lammps_data_path = os.path.join(output_folder_path, 'data.data')
lammps_input_path = os.path.join(output_folder_path, 'input.in')
lammps_output_path = os.path.join(output_folder_path, 'dump_y.stress')
ovito_image_path = os.path.join(output_folder_path, 'final_image.png')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lammps')
def select_image():
    ai_images, real_images = get_images()
    selected_ai_images = random.sample(ai_images, 8)
    selected_real_images = random.sample(real_images, 8)
    images = selected_ai_images + selected_real_images
    random.shuffle(images)
    return render_template('animalImageLammp.html', images=images)

@app.route('/process_image', methods=['POST'])
def process_image():
    selected_image = request.form.get('selected_image')
    image_path = os.path.join('static', selected_image)
    
    # Define the command to run the external Python script
    script_command = ['python', 'lammps.py', image_path]
    
    # Run the script and wait for it to complete
    result = subprocess.run(script_command, capture_output=True, text=True)
    
    # Check if the script ran successfully
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    
    return redirect(url_for('show_image'))

@app.route('/show_image')
def show_image():
    # Assuming 'outputImage/final_image.png' exists in your static folder binary_image
    image_path = 'outputImage/final_image.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/binary_image')
def binary_image():
    # Assuming 'outputImage/final_image.png' exists in your static folder binary_image
    image_path = 'outputImage/binary_image.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/pattern_generator')
def pattern_generator():
    return render_template('pattern_generator.html')

@app.route('/generate_zebra_pattern')
def generate_zebra_pattern():
    size = 128
    Du = float(request.args.get('Du', 0.16))
    Dv = float(request.args.get('Dv', 0.08))
    F = float(request.args.get('F', 0.035))
    k = float(request.args.get('k', 0.060))

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
    try:
        selections = {}
        responses = []
        userID = str(uuid4())
        user_age = request.form.get('age')
        for img in request.form:
            selection = request.form[img]
            selections[img] = selection
            image_type = 1 if 'real' in img else 0  # 1 for real, 0 for AI
            answer = 1 if selection == 'real' else 0  # 1 for real, 0 for AI
            correct = 1 if answer == image_type else 0
            responses.append({
                'ID': userID,
                'UserAge': user_age,
                'Image Name': img,
                'AI/Real': image_type,
                'Answer': answer,
                'Correct': correct
            })

        df = pd.DataFrame(responses)

        # Define the file path
        file_path = '/home/avjadhav/server/Zoo_server/responses.csv'

        # for testing local 
        # file_path = 'responses.csv'

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)  # Changed to read_csv
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(file_path, index=False)
        return render_template('responseguess.html', responses=responses, selections=selections)
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


def get_images():
    static_dir = 'static'
    ai_images = [img for img in os.listdir(static_dir) if img.startswith('ai_images')]
    real_images = [img for img in os.listdir(static_dir) if img.startswith('real_images')]
    return ai_images, real_images


# @app.route('/process_image')
# def process_image():
#     try:
#         # Assume image_path is a path to a static image for testing purposes
#         image_path = 'static/real_images51.jpg'
        
#         output_folder_path = 'outputImage'
#         binary_image_path = os.path.join(output_folder_path, 'binary_image.png')
#         lammps_data_path = os.path.join(output_folder_path, 'data.data')
#         lammps_input_path = os.path.join(output_folder_path, 'input.in')
#         lammps_output_path = os.path.join(output_folder_path, 'dump_y.stress')
#         ovito_image_path = os.path.join(output_folder_path, 'final_image.png')
        
#         if not os.path.exists(output_folder_path):
#             os.makedirs(output_folder_path)

#         lammps.generate_model(image_path, output_folder_path, binary_image_path, lammps_data_path)
        
#         lammps.write_lammps_input(lammps_input_path, lammps_data_path)
        
#         lammps.run_lammps_simulation(lammps_input_path, output_folder_path)
        
#         lammps.create_image_from_lammps_output(lammps_output_path, ovito_image_path)
        
#         return send_file(ovito_image_path, mimetype='image/png')

#     except Exception as e:
#         app.logger.error(f"Error processing image: {str(e)}")
#         return jsonify({"error": str(e)}), 500


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
