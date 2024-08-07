import os
import sys
import argparse
from PIL import Image
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import cKDTree
import subprocess
from PySide6.QtWidgets import QApplication
from ovito.io import import_file, export_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import ColorCodingModifier
from ovito.io import import_file
from ovito.modifiers import AssignColorModifier, ColorCodingModifier, ExpressionSelectionModifier
from ovito.vis import Viewport, TachyonRenderer
import math
from PIL import Image
import math

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process an image to generate a LAMMPS model.')
parser.add_argument('image_path', type=str, help='Path to the input image')
parser.add_argument('output_user_folder', type=str, help='Path to the output_user_folder image')
args = parser.parse_args()

# Paths
image_path = args.image_path
output_folder_path = args.output_user_folder
binary_image_path = os.path.join(output_folder_path, 'binary_image.png')
lammps_data_path = os.path.join(output_folder_path, 'data.data')
lammps_input_path = os.path.join(output_folder_path, 'input.in')
lammps_output_path = os.path.join(output_folder_path, 'dump_y.stress')
ovito_image_path = os.path.join(output_folder_path, 'configuration.png')
stress_field_path = os.path.join(output_folder_path, 'stress_field.png')
combined_image_path = os.path.join(output_folder_path, 'stitched_images.png')
state_file_path = os.path.join(output_folder_path, 'state.txt')

# Create output directories if they don't exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(os.path.dirname(binary_image_path), exist_ok=True)

# Initialize QApplication for offscreen rendering
app = QApplication(sys.argv)
if app is None:
        app = QApplication(sys.argv)

# Function to update state file
def update_state(state_file_path, message):
    with open(state_file_path, 'w') as state_file:
        state_file.write(message)

def convert_and_invert_binary_image(image_path, output_path, threshold=128):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    binary_img_array = np.where(img_array >= threshold, 1, 0)

    inverted_img_array = np.where(binary_img_array == 0, 255, 0).astype(np.uint8)

    inverted_img = Image.fromarray(inverted_img_array)
    inverted_img.save(output_path)


# Step 1: Image Processing and Model Generation
def generate_model(image_path, output_folder_path, binary_image_path, lammps_data_path):
    update_state(state_file_path, "Step 1: Image Processing and Model Generation")
    img = Image.open(image_path).convert('L')
    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    grayImage = np.array(img_resized)

    # Convert grayscale to binary image using a threshold
    binaryImage = np.where(grayImage >= 128, 1, 0)

    cmap = ListedColormap(['white', 'black'])
    plt.imshow(binaryImage, cmap=cmap)
    plt.axis('off')
    plt.savefig(binary_image_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    a = 0.373  # Lattice constant
    lx = binaryImage.shape[1] * a
    ly = binaryImage.shape[0] * a
    nx = int(lx / a)
    ny = int(ly / (a / 2 * sqrt(3)))
    xx = []
    natom = 0
    nb = 0

    # Generate atom positions
    for j in range(ny):
        for i in range(nx):
            x = (i * a + (j % 2) * a / 2)
            y = (j * a / 2 * sqrt(3))
            map_y = int(x / lx * binaryImage.shape[1])
            map_x = binaryImage.shape[0] - 1 - int(y / ly * binaryImage.shape[0])  # Correcting the mapping
            atom_type = binaryImage[map_x, map_y] + 1
            xx.append([atom_type, x, y, 0.0])
            natom += 1

    bonds = []
    a_cutoff = 1.1 * a
    positions = np.array(xx)[:, 1:4]
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=a_cutoff, output_type='ndarray')

    for pair in pairs:
        i, j = pair
        bond_type = 3 if xx[i][0] != xx[j][0] else xx[i][0]
        bonds.append([bond_type, i + 1, j + 1])
        nb += 1

    # Writing to LAMMPS data file
    with open(lammps_data_path, 'w') as data_file:
        data_file.write('LAMMPS data file\n\n')
        data_file.write(f'{natom} atoms\n')
        data_file.write(f'{nb} bonds\n')
        data_file.write('2 atom types\n3 bond types\n\n')
        xlo_xhi = min(positions[:,0]) - (max(positions[:,0]) - min(positions[:,0])) * 0.1, max(positions[:,0]) + (max(positions[:,0]) - min(positions[:,0])) * 0.1
        ylo_yhi = min(positions[:,1]) - (max(positions[:,1]) - min(positions[:,1])) * 0.1, max(positions[:,1]) + (max(positions[:,1]) - min(positions[:,1])) * 0.1
        zlo_zhi = min(positions[:,2]) - (max(positions[:,0]) - min(positions[:,0])) * 0.2, max(positions[:,2]) + (max(positions[:,0]) - min(positions[:,0])) * 0.2
        data_file.write(f'{xlo_xhi[0]} {xlo_xhi[1]} xlo xhi\n')
        data_file.write(f'{ylo_yhi[0]} {ylo_yhi[1]} ylo yhi\n')
        data_file.write(f'{zlo_zhi[0]} {zlo_zhi[1]} zlo zhi\n\n')
        data_file.write('Masses\n\n')
        data_file.write('1 1e4\n2 1e4\n\n')
        data_file.write('Atoms\n\n')
        for i, atom in enumerate(xx):
            data_file.write(f'{i+1} 1 {atom[0]} {atom[1]} {atom[2]} {atom[3]}\n')
        data_file.write('\nBonds\n\n')
        for i, bond in enumerate(bonds):
            data_file.write(f'{i+1} {bond[0]} {bond[1]} {bond[2]}\n')

# Step 2: Writing LAMMPS Input Script
def write_lammps_input(lammps_input_path, lammps_data_path):
    update_state(state_file_path, "Step 2: Writing LAMMPS Input Script")
    lammps_input_content = f"""
   ################################################
   # INPUT FILE
   ################################################
   # Deform a single layer metal polymer composite film
   # Zhao Qin 2023 @ SU
   ################################################
   # Units, angle, tensile parameters etc.
   ################################################
   boundary p p p
   units       micro
   atom_style      bond
   timestep    0.002
   dimension       3
   read_data   {lammps_data_path}
   neighbor    0.2 bin
   neigh_modify    every 100 delay 100
   pair_style soft 0.8
   pair_coeff * * 1e7 0.8
   bond_style      morse
   bond_coeff      1 1028.2 25.13 0.373
   bond_coeff      2 268.7 4.49 0.373
   bond_coeff      3 648.45 14.81 0.373
   ################################################
   # Boundary conditions 
   ################################################
   region          1 block INF INF INF 2.488 INF  INF units box
   group           lower region 1
   region          2 block   INF INF  93  INF INF INF units box    
   group           upper region 2
   region          3 block   INF 2.488 INF INF INF INF units box
   group           left  region 3
   region          4 block   93 INF INF INF INF INF units box
   group           right region 4
   group           boundary union lower upper
   group           mobile subtract all boundary
   min_style       cg
   min_modify      dmax 0.01
   minimize        0.0 0.0 200 200
   compute         peratom all stress/atom NULL
   fix 999 all ave/atom 1 10 10 c_peratom[1] c_peratom[1] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
   variable mises atom sqrt((f_999[1]-f_999[2])*(f_999[1]-f_999[2])+(f_999[2]-f_999[3])*(f_999[2]-f_999[3])+(f_999[1]-f_999[3])*(f_999[1]-f_999[3])+6*(f_999[4]*f_999[4]+f_999[5]*f_999[5]+f_999[6]*f_999[6]))
   dump 400 all custom 100 {lammps_output_path} type x y z v_mises f_999[1] f_999[2] f_999[3] f_999[4] f_999[5] f_999[6]
   velocity        all create 300.00 376847
   fix             initnve         all nve
   fix             initcont  all temp/rescale 100 300.0 300.0  10.  0.5
   thermo          100
   velocity        upper set NULL 0.0 NULL units box
   velocity        lower set NULL 0.0 NULL units box
   fix             2 upper setforce NULL 0.00 NULL
   fix             21 lower setforce NULL 0.00 NULL
   thermo_style    custom step temp etotal
   thermo_modify   flush yes 
   ################################################
   # Relax structure for 100 steps
   ################################################
   run     1000
   # NOW: One-step tensile deformation applied
    ################################################
    fix         stretch all deform 1 y delta 0.0 5.0 units box
    thermo_style    custom step f_2[2] f_21[2] temp etotal
    thermo_modify   flush yes
    thermo          100
    run             10
    ################################################
    # Relax structure after deformation
    ################################################
    #minimize        0.0 0.0 2000 2000
    thermo_style    custom step f_2[2] f_21[2] temp etotal
    thermo_modify   flush yes
    run             2000
    """
    with open(lammps_input_path, 'w') as f:
        f.write(lammps_input_content)

# Step 3: Running LAMMPS Simulation
def run_lammps_simulation(lammps_input_path, output_folder_path):
    update_state(state_file_path, "Step 3: Running Lammps Simulation")
    run_script = f"""
    lmp -in {lammps_input_path} > {output_folder_path}/indent_out_y.out
    """
    run_script_path = os.path.join(output_folder_path, 'run.sh')
    with open(run_script_path, 'w') as f:
        f.write(run_script)
    os.chmod(run_script_path, 0o755)
    subprocess.run(run_script_path, shell=True)

def create_image_from_lammps_output(lammps_output_path, ovito_image_path, stress_field_path, combined_image_path):
    update_state(state_file_path, "Step 4: Creating Images from LAMMPS Output")
    # Load the LAMMPS output dump file into OVITO
    pipeline = import_file(lammps_output_path, multiple_frames=True)
    
    # Access the last frame of the imported animation sequence
    last_frame = pipeline.source.num_frames - 1
    data = pipeline.compute(last_frame)
    
    # Add pipeline to scene
    pipeline.add_to_scene()
    
    # Select particles of type 1 and assign white color
    selection1 = ExpressionSelectionModifier(expression='ParticleType == 1')
    pipeline.modifiers.append(selection1)
    color_modifier1 = AssignColorModifier(color=(1, 1, 1))  # White color
    pipeline.modifiers.append(color_modifier1)

    # Select particles of type 2 and assign black color
    selection2 = ExpressionSelectionModifier(expression='ParticleType == 2')
    pipeline.modifiers.append(selection2)
    color_modifier2 = AssignColorModifier(color=(0, 0, 0))  # Black color
    pipeline.modifiers.append(color_modifier2)

    # Recompute to apply color modifiers
    data = pipeline.compute()

    # Setup viewport for rendering
    vp = Viewport()
    vp.type = Viewport.Type.Ortho
    vp.camera_pos = (122.475, 254.5, 0)
    vp.camera_dir = (0, 0, -1)
    vp.fov = math.radians(16194)

    # Render the configuration image
    vp.render_image(size=(2810, 2810), filename=ovito_image_path, background=(1, 1, 1), frame=last_frame, crop=True, renderer=TachyonRenderer())
    print(f'Configuration image saved to {ovito_image_path}')

    # Remove color modifiers
    pipeline.modifiers.clear()

    # Add stress visualization
    pipeline.modifiers.append(ColorCodingModifier(property='v_mises', gradient=ColorCodingModifier.Hot(), start_value=0, end_value=20000))

    # Recompute to apply stress visualization
    data = pipeline.compute()

    # Render the stress field image
    vp.render_image(size=(2810, 2810), filename=stress_field_path, background=(1, 1, 1), frame=last_frame, crop=True, renderer=TachyonRenderer())
    print(f'Stress field image saved to {stress_field_path}')

    # Clean up: remove pipeline from scene and release resources
    pipeline.remove_from_scene()
    del pipeline

    # Stitch the  images side by side
    image1 = Image.open(binary_image_path)
    image2 = Image.open(ovito_image_path)
    image3 = Image.open(stress_field_path)
    # Resize images to have the same width
    target_width = max(image1.width, image2.width, image3.width)
    image1 = image1.resize((target_width, int(image1.height * (target_width / image1.width))))
    image2 = image2.resize((target_width, int(image2.height * (target_width / image2.width))))
    image3 = image3.resize((target_width, int(image3.height * (target_width / image3.width))))

    # Create a new image with the combined width and maximum height of the images
    combined_width = target_width * 3
    combined_height = max(image1.height, image2.height, image3.height)
    
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the combined image
    combined_image.paste(image1, (0, int((image2.height - image1.height) / 2)))
    combined_image.paste(image2, (image1.width, 0))
    combined_image.paste(image3, (image1.width + image2.width, 0))
    combined_image = combined_image.crop((0, 0, combined_width, combined_height))


    # Save the combined image
    combined_image.save(combined_image_path)
    print(f'Combined image saved to {combined_image_path}')

# Main workflow
generate_model(image_path, output_folder_path, binary_image_path, lammps_data_path)
write_lammps_input(lammps_input_path, lammps_data_path)
run_lammps_simulation(lammps_input_path, output_folder_path)
create_image_from_lammps_output(lammps_output_path, ovito_image_path,stress_field_path,combined_image_path)
update_state(state_file_path, "Completed")
print(f'Final image saved to {ovito_image_path}')
