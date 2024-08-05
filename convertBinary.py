import os
from PIL import Image
import numpy as np

def convert_and_invert_binary_image(image_path, output_path, threshold=128):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    binary_img_array = np.where(img_array >= threshold, 1, 0)

    inverted_img_array = np.where(binary_img_array == 0, 255, 0).astype(np.uint8)

    inverted_img = Image.fromarray(inverted_img_array)
    inverted_img.save(output_path)

def process_images_in_directory(input_dir, output_dir, threshold=128):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            convert_and_invert_binary_image(input_path, output_path, threshold)
            print(f'Processed {filename}')

input_directory = 'static/zebra'
output_directory = 'static/zebra_inverse'

process_images_in_directory(input_directory, output_directory)