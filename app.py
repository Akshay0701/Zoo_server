import random
import subprocess
from uuid import uuid4
import uuid
from flask import Flask, render_template, send_file, jsonify, request, redirect, send_from_directory, url_for
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import pandas as pd
import logging
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from projectideacreation import NSFProjectChain
from research_extractor import extract_research_interests  # Assuming this exists
from team_creator import form_teams, extract_main_research_areas

app = Flask(__name__)
app.config['TIMEOUT'] = 90  # Example: 90 seconds
CORS(app)  # Allows all origins by default

@app.route('/')
def index():
    return render_template('index.html')

def process_image_task(image_path, output_user_folder):
    script_command = ['python3', 'lammps.py', image_path, output_user_folder]
    subprocess.run(script_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@app.route('/get_animal_images', methods=['GET'])
def get_animal_images():
    animal = request.args.get('animal')
    folder_path = os.path.join('static', animal)
    
    # Get all images and limit to the first 16
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    images = images[:16]  # Limit to the first 16 images

    return render_template('select_image_lammps.html', animal=animal, images=images)

@app.route('/process_image', methods=['POST'])
def process_image():
    selected_image = request.form.get('selected_image')
    animal_choice = request.form.get('animal_choice')
    image_path = os.path.join('static', animal_choice, selected_image)

    user_folder = str(uuid.uuid4())
    output_user_folder = os.path.join('outputImage', user_folder)
    os.makedirs(output_user_folder, exist_ok=True)

    # Start the task in a new thread
    thread = threading.Thread(target=process_image_task, args=(image_path, output_user_folder))
    thread.start()

    # Redirect to a route that will handle showing the image after processing
    return redirect(url_for('show_image', user_folder=user_folder))

@app.route('/show_image/<user_folder>')
def show_image(user_folder):
    return render_template('show_image.html', user_folder=user_folder)

@app.route('/check_image/<user_folder>')
def check_image(user_folder):
    required_images = ['configuration.png', 'stitched_images.png', 'stress_field.png']
    image_dir = os.path.join('outputImage', user_folder)
    
    for image in required_images:
        image_path = os.path.join(image_dir, image)
        if not os.path.exists(image_path):
            return 'Step 1: Image Processing and Model Generation'
    
    return 'ready'

@app.route('/get_status/<user_folder>')
def get_status(user_folder):
    state_file_path = os.path.join('outputImage', user_folder, 'state.txt')
    if not os.path.exists(state_file_path):
        return 'Step 1: Image Processing and Model Generation'
    with open(state_file_path, 'r') as state_file:
        status = state_file.read().strip()
    return status

@app.route('/get_image/<user_folder>')
def get_image(user_folder):
    return send_from_directory(os.path.join('outputImage', user_folder), 'configuration.png')

@app.route('/get_ovito_image/<user_folder>')
def get_ovito_image(user_folder):
    return send_from_directory(os.path.join('outputImage', user_folder), 'stitched_images.png')

@app.route('/get_stress_image/<user_folder>')
def get_stress_image(user_folder):
    return send_from_directory(os.path.join('outputImage', user_folder), 'stress_field.png')


@app.route('/lammps')
def lammps():
    ai_images, real_images = get_images()
    selected_ai_images = random.sample(ai_images, 8)
    selected_real_images = random.sample(real_images, 8)
    images = selected_ai_images + selected_real_images
    random.shuffle(images)
    return render_template('animalImageLammp.html', images=images)

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
            if img == 'age':
                continue
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

def get_images():
    static_dir = 'static'
    ai_images = [img for img in os.listdir(static_dir) if img.startswith('ai_images')]
    real_images = [img for img in os.listdir(static_dir) if img.startswith('real_images')]
    return ai_images, real_images


# Existing endpoint (example, adjust as per your actual implementation)
@app.route('/extract_interests', methods=['POST'])
def extract_interests():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    results = []
    for profile in data:
        name = profile.get('name')
        description = profile.get('description')
        if not name or not description:
            continue
        topics = extract_research_interests(description)
        results.append({"name": name, "research_topics": topics})
    return jsonify(results), 200

# New endpoint for team creation
@app.route('/teamcreation', methods=['POST'])
def teamcreation():
    """
    Creates teams from a list of researchers based on their research topics.
    
    Expects a JSON payload like:
    [
        {"name": "John Doe", "research_topics": ["Machine Learning", "NLP"]},
        {"name": "Jane Smith", "research_topics": ["Computer Vision", "Robotics"]}
    ]
    
    Returns a JSON response with team details.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Convert input list to dictionary
    profiles = {profile['name']: profile['research_topics'] for profile in data}

    # Form teams
    teams = form_teams(profiles, max_team_size=4)

    # Extract research areas
    team_research_areas = extract_main_research_areas(profiles, teams)

    # Structure the output
    output = []
    for team_id, members in enumerate(teams):
        team_data = {
            "team_id": team_id + 1,
            "team_size": len(members),
            "members": members,
            "team_research_areas": team_research_areas[team_id]["team_fields"],
            "member_fields": team_research_areas[team_id]["member_fields"]
        }
        output.append(team_data)

    return jsonify(output)


# Instantiate the NSFProjectChain
nsf_chain = NSFProjectChain()

@app.route('/generate-proposals', methods=['POST'])
def generate_proposals():
    teams = request.get_json()
    if not teams:
        return jsonify({"error": "No teams data provided"}), 400
    
    # For each team, call Groq via our NSFProjectChain to generate project proposals.
    for team in teams:
        try:
            proposals = nsf_chain.generate_project_proposals(team)
            team["project_proposals"] = proposals
        except Exception as e:
            # Optionally log the error; here we include an error message in the team output.
            team["project_proposals"] = [f"Error generating proposals: {str(e)}"]
    
    # Return the updated teams list as JSON.
    return jsonify(teams), 200


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)



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
