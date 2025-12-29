import numpy as np
import os
import pylab as plt
import matplotlib
from PIL import Image
from pathlib import Path
import random
import argparse
matplotlib.use('TkAgg')

# --- aggiungi parsing degli argomenti ---
parser = argparse.ArgumentParser(description="Limit distance maps to threshold and save RGBA images.")
parser.add_argument('--distance_maps_folder', type=str, required=True, help='Path to distance maps folder')
parser.add_argument('--images_folder', type=str, required=True, help='Path to images folder')
parser.add_argument('--save_folder', type=str, required=True, help='Path to save thresholded images')
parser.add_argument('--threshold', type=float, required=False, help='Threshold value for depth (optional)')
args = parser.parse_args()

distance_maps_folder = args.distance_maps_folder
images_folder = args.images_folder
save_folder = args.save_folder

# Ottieni tutti i file delle immagini
all_image_files = []
for subdir, dirs, files in os.walk(images_folder):
    for file in files:
        all_image_files.append((subdir, file))

# --- threshold logic ---
if args.threshold is not None:
    threshold = args.threshold
else:
    # Scegli 3 immagini random
    random_samples = random.sample(all_image_files, min(3, len(all_image_files)))

    for subdir, file in random_samples:
        image_path = os.path.join(subdir, file)
        image_filename_base = Path(file).stem

        depth_filename = image_filename_base + "_distance.npz"
        depth_path = os.path.join(distance_maps_folder, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Distance map not found for image {file}, skipping.")
            continue

        depth_map_file = np.load(depth_path)
        depth_map = depth_map_file['distance']

        plt.imshow(depth_map, cmap='viridis')
        plt.title("Metric Distance Map")
        plt.show()

    #ask after showing the first 3 distance maps
    print("Enter the threshold value for depth: ", flush=True)
    threshold = float(input())

print(f"__THRESHOLD__:{threshold}", flush=True)

save_folder = save_folder + "_" + str(threshold)
os.makedirs(save_folder, exist_ok=True)


# Loop on all images
for subdir, dirs, files in os.walk(images_folder):
    for file in files:
        
        image_path = os.path.join(subdir, file)
        image_filename_base = Path(file).stem

        depth_filename = image_filename_base + "_distance.npz"
        depth_path = os.path.join(distance_maps_folder, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Distance map not found for image {file}, skipping.")
            continue

        depth_map_file = np.load(depth_path)
        depth_map = depth_map_file['distance']
        
        # Create a copy of the depth map array
        binary_mask = np.copy(depth_map)

        # ------------------------------

        # *********** FOR BACKGROUND
        # INCLUDE ALL PIXELS WITH DEPTH GREATER OR EQUAL THAN THE THRESHOLD

        # Create a binary mask 
        binary_mask[binary_mask < threshold] = 0.0
        binary_mask[binary_mask >= threshold] = 1.0 # cioè ci va bene anche la threshold stessa

        # -------------------------------
        '''
        # *********** FOR FOREGROUND
        # INCLUDE ALL PIXELS WITH DEPTH LESS THAN THE THRESHOLD

        # Create a binary mask 
        binary_mask[binary_mask < threshold] = 1.0
        binary_mask[binary_mask >= threshold] = 0.0 # cioè escludo anche la threshold stessa
        '''
        # -------------------------------
        
        '''
        plt.imshow(binary_mask, cmap='gist_gray')
        plt.title("Binary Mask")
        plt.show()
        '''
        

        '''
        # Save the binary mask

        #remove until char _
        save_filename = file.split("_")[0] + "_binary.npy"
        save_path = os.path.join(subdir, save_filename)
        np.save(save_path, binary_mask)
        '''

        # Carica l'immagine JPG
        image = Image.open(image_path).convert("RGB")  # Assicura che sia RGB
        image_np = np.array(image)  # Converti in NumPy array (HxWx3)

        # Assicurati che la maschera sia nel range corretto (0-255)
        alpha_channel = (binary_mask * 255).astype(np.uint8)

        # Converti l'immagine in RGBA aggiungendo il canale alfa
        image_rgba = np.dstack((image_np, alpha_channel))  #  Diventa HxWx4 (RGBA)

        # Crea un'immagine PIL RGBA
        image_rgba_pil = Image.fromarray(image_rgba)

        '''
        # mostra immagine tagliata
        plt.imshow(image_rgba)
        plt.title("Img")
        plt.show()
        '''

        # Salva l'immagine con trasparenza
        save_filename = file
        save_path = os.path.join(save_folder, save_filename)
        image_rgba_pil.save(save_path)
