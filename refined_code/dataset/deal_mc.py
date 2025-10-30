"""
deal_mc.py - Part of fNIR Base Model.
"""

import os
import numpy as np
import scipy.io
import h5py
from scipy.interpolate import griddata
from tqdm import tqdm  # Import tqdm for progress bar

# Function to load .mat files using scipy.io.loadmat
def load_mat_with_scipy(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        # List the contents of the file
        print(f"Keys in the .mat file (scipy): {list(data.keys())}")
        # Assuming 'dynamic_mua' and 'node' are in the .mat file
        absorption_coeff = np.array(data['dynamic_mua'])  # Absorption coefficient
        xyz_coords = np.array(data['node'])  # XYZ coordinates
        return absorption_coeff, xyz_coords
    except NotImplementedError:
        # If scipy.io.loadmat cannot handle this file (e.g., due to version), fallback to h5py
        print(f"scipy.io.loadmat failed for {file_path}. Trying h5py...")
        return load_mat_with_h5py(file_path)
    except ValueError as e:
        # Handle other errors, such as corrupted files
        print(f"Error loading {file_path} with scipy.io.loadmat: {e}")
        return None, None

# Function to load .mat files using h5py
def load_mat_with_h5py(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Keys in the .mat file (h5py): {list(f.keys())}")
            # Assuming 'dynamic_mua' and 'node' are in the .mat file
            absorption_coeff = np.array(f['dynamic_mua'])
            xyz_coords = np.array(f['node'])
        return absorption_coeff, xyz_coords
    except Exception as e:
        print(f"Error loading {file_path} with h5py: {e}")
        return None, None


# Function to normalize absorption coefficient for each file individually



# Function to generate height-divided data slices for each time step
def generate_data_slices(file_path, height_slices=9):
    # Load .mat file
    absorption_coeff, xyz_coords = load_mat_with_scipy(file_path)

    # If loading failed, skip further processing
    if absorption_coeff is None or xyz_coords is None:
        return None

    # Extract unique time steps
    num_time_steps = absorption_coeff.shape[1]

    # Initialize an array to store the rotated grid absorptions in 4D (102, 128, height_slices, num_time_steps)
    rotated_grid_absorptions = np.zeros((102, 128, height_slices, num_time_steps))

    # For each time step, generate data for each of the 9 height slices
    for t in range(num_time_steps):
        absorption_values = absorption_coeff[:, t]  # Absorption at time step 't'

        # Generate a grid for plotting
        grid_x, grid_y, grid_z = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]

        # Determine min and max z values to divide into slices
        min_z, max_z = np.min(grid_z), np.max(grid_z)
        z_slices = np.linspace(min_z, max_z, height_slices + 1)

        # For each height slice, generate the corresponding grid and store the result
        for i in range(height_slices):
            # Mask to select the points within the current height slice
            mask = (grid_z >= z_slices[i]) & (grid_z < z_slices[i + 1])
            if np.sum(mask) == 0:
                continue  # Skip if no points in the current slice

            # Extract the coordinates and absorption values within the current slice
            slice_x = grid_x[mask]
            slice_y = grid_y[mask]
            slice_absorption = absorption_values[mask]

            # Generate a grid for plotting with fixed resolution of 102x128
            grid_xi, grid_yi = np.mgrid[
                               min(slice_x):max(slice_x):128j,
                               min(slice_y):max(slice_y):102j
                               ]

            # Interpolate absorption values onto the grid
            grid_absorption = griddata(
                (slice_x, slice_y), slice_absorption, (grid_xi, grid_yi), method='cubic'
            )

            # Rotate the image 90 degrees to the right
            rotated_grid_absorption = np.rot90(grid_absorption, k=-1)  # Rotate 90 degrees clockwise

            # Handle NaN values by replacing them with zero
            rotated_grid_absorption = np.nan_to_num(rotated_grid_absorption, nan=0.0)

            # Store the rotated absorption grid in the appropriate slot in the 4D array
            rotated_grid_absorptions[:, :, i, t] = rotated_grid_absorption

    return rotated_grid_absorptions

# Function to process all .mat files in a directory and save results as npy
# Function to process all .mat files in a directory and save results as npy
def process_and_save_all_mat_files(directory_path, save_path, height_slices=9):
    results = {}  # Store data for each file in a dictionary
    all_filenames = []  # Store filenames

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get all .mat files
    mat_files = [f for f in os.listdir(directory_path) if f.endswith(".mat")]

    # Loop through and process each .mat file
    for filename in tqdm(mat_files, desc="Processing .mat files", unit="file"):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {file_path}...")

        # Check if the file actually exists before processing
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}. Skipping.")
            continue

        # Get rotated_grid_absorption data
        rotated_grid_absorptions = generate_data_slices(file_path, height_slices)

        if rotated_grid_absorptions is None:
            print(f"Skipping {filename} due to loading error.")
            continue

        # Save each file's data in the dictionary
        results[filename] = rotated_grid_absorptions
        all_filenames.append(filename)

    # Save the dictionary as a .npy file
    np.save(os.path.join(save_path, 'datas.npy'), results)

    print(f"Saved results to {save_path}")

# Example usage
directory_path = 'data/source data/fdDOT_DynamicMC_fix_ellips v2'  # .mat 文件目录
save_path = 'data/source data/fdDOT_DynamicMC_fix_ellips v2'  # 保存结果的目录


process_and_save_all_mat_files(directory_path, save_path)

