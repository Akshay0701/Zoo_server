import os
import sys
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import cKDTree
import subprocess
from PySide6.QtWidgets import QApplication
from ovito.io import import_file, export_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import ColorCodingModifier
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
ovito_image_path = os.path.join(output_folder_path, 'final_image.png')

# Create output directories if they don't exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(os.path.dirname(binary_image_path), exist_ok=True)

# Initialize QApplication for offscreen rendering
app = QApplication(sys.argv)

# Step 1: Image Processing and Model Generation
def generate_model(image_path, output_folder_path, binary_image_path, lammps_data_path):
    img = Image.open(image_path).convert('L')
    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    grayImage = np.array(img_resized)

    # Convert grayscale to binary image using a threshold
    binaryImage = np.where(grayImage >= 50, 1, 0)
    plt.imshow(binaryImage, cmap='gray')
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
            map_y = int(i / nx * binaryImage.shape[1])
            map_x = int(j / ny * binaryImage.shape[0])
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
        xlo_xhi = min(positions[:,0]) - (max(positions[:,0]) - min(positions[:,0])) * 1, max(positions[:,0]) + (max(positions[:,0]) - min(positions[:,0])) * 1
        ylo_yhi = min(positions[:,1]) - (max(positions[:,1]) - min(positions[:,1])) * 0.5, max(positions[:,1]) + (max(positions[:,1]) - min(positions[:,1])) * 0.5
        zlo_zhi = min(positions[:,2]) - (max(positions[:,0]) - min(positions[:,0])) * 1, max(positions[:,2]) + (max(positions[:,0]) - min(positions[:,0])) * 1
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
    units		micro
    atom_style      bond
    timestep	0.002
    dimension       3
    read_data	{lammps_data_path}
    neighbor	0.2 bin
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
    region          4 block   INF INF 93 INF INF INF units box
    group           right region 4
    group           boundary union lower upper
    group           mobile subtract all boundary
    min_style       cg
    min_modify      dmax 0.01
    minimize        0.0 0.0 2000 2000
    compute         peratom all stress/atom NULL
    fix 999 all ave/atom 10 5 100 c_peratom[1] c_peratom[1] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
    variable mises atom sqrt((f_999[1]-f_999[2])*(f_999[1]-f_999[2])+(f_999[2]-f_999[3])*(f_999[2]-f_999[3])+(f_999[1]-f_999[3])*(f_999[1]-f_999[3])+6*(f_999[4]*f_999[4]+f_999[5]*f_999[5]+f_999[6]*f_999[6]))
    dump 400 all custom 500 {lammps_output_path} type x y z v_mises f_999[1] f_999[2] f_999[3] f_999[4] f_999[5] f_999[6]
    velocity        all create 300.00 376847
    fix             initnve         all nve
    fix             initcont  all temp/rescale 100 300.0 300.0  10.  0.5
    thermo          100
    velocity        upper set NULL 0.0 NULL units box
    velocity        lower set NULL 0.0 NULL units box
    fix             2 upper setforce NULL 0.00 NULL
    fix             21 lower setforce NULL 0.00 NULL
    thermo_style	custom step temp etotal
    thermo_modify   flush yes 
    ################################################
    # Relax structure for 200000 steps
    ################################################
    run		10
    ################################################
    # NOW: Loading loop applied
    ################################################
    ##unfix wallhi
    compute         max_y all reduce max y
    compute         min_y all reduce min y
    fix      stretch all deform 100 y erate 0.0001
    thermo_style    custom step f_2[2] f_21[2] c_max_y c_min_y temp etotal
    thermo_modify   flush yes
    thermo          1000
    ###############################################m
    # Decide at which atom you pull (change
    #     when you change 'NL' in 'long_pull.py')
    ################################################
    run              2000
    """
    with open(lammps_input_path, 'w') as f:
        f.write(lammps_input_content)

# Step 3: Running LAMMPS Simulation
def run_lammps_simulation(lammps_input_path, output_folder_path):
    run_script = f"""
    #!/bin/bash
    #
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --job-name=zebra_yy
    #SBATCH --time=10:00:00
    #SBATCH --output=job_out
    #SBATCH --error=job_err
    export PATH=/usr/bin:$PATH
    lmp -in {lammps_input_path} > {output_folder_path}/indent_out_y.out
    """
    run_script_path = os.path.join(output_folder_path, 'run.sh')
    with open(run_script_path, 'w') as f:
        f.write(run_script)
    os.chmod(run_script_path, 0o755)
    subprocess.run(run_script_path, shell=True)

def create_image_from_lammps_output(lammps_output_path, ovito_image_path):
    # Ensure the LAMMPS output file exists and is valid
    if not os.path.exists(lammps_output_path):
        print(f"Error: {lammps_output_path} does not exist.")
        return
    
    # Load the LAMMPS output dump file into OVITO
    pipeline = import_file(lammps_output_path, multiple_frames=True)
    
    # Ensure the pipeline has frames
    if pipeline.source.num_frames == 0:
        print("Error: No frames in the pipeline.")
        return
    
    # Access the last frame of the imported animation sequence
    last_frame = pipeline.source.num_frames - 1
    data = pipeline.compute(last_frame)
    
    # Ensure data is not empty
    if data.particles.count == 0:
        print("Error: No particles in the data.")
        return
    
    # Add pipeline to scene
    pipeline.add_to_scene()
    
    # Modify particle visualization settings
    pipeline.modifiers.append(ColorCodingModifier(property='v_mises', gradient=ColorCodingModifier.Hot(), only_selected=True, start_value=0, end_value=200))
    
    # Setup viewport for rendering
    vp = Viewport()
    vp.type = Viewport.Type.Ortho
    vp.camera_pos = (122.475, 254.5, 0)
    vp.camera_dir = (0, 0, -1)
    vp.fov = math.radians(16194)
    
    # Render the image
    vp.render_image(size=(2810, 2810), filename=ovito_image_path, background=(1, 1, 1), frame=last_frame, crop=True, renderer=TachyonRenderer())
    
    # Clean up: remove pipeline from scene and release resources
    pipeline.remove_from_scene()
    del pipeline
    
    print(f'Image saved to {ovito_image_path}')

# Main workflow
generate_model(image_path, output_folder_path, binary_image_path, lammps_data_path)
write_lammps_input(lammps_input_path, lammps_data_path)
run_lammps_simulation(lammps_input_path, output_folder_path)
create_image_from_lammps_output(lammps_output_path, ovito_image_path)

print(f'Final image saved to {ovito_image_path}')
