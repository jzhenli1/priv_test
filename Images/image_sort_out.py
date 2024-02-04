import os
import pandas as pd

# Read the Excel file
df = pd.read_csv('sampled_edges.csv', sep=';', header=None)
# Assuming the image names are in the first column
subset_images = df[df.columns[2]].tolist()

main_folder = 'edges'
subset_folder = 'survey_edges'

for img_name in subset_images:
    main_image_path = os.path.join(main_folder, img_name)
    subset_image_path = os.path.join(subset_folder, img_name)
    
    # Check if the image exists in both main and subset folders
    if os.path.exists(main_image_path) and os.path.exists(subset_image_path):
        # Remove or move the image from the main folder
        os.remove(main_image_path)  # Use this to delete the file
        # shutil.move(main_image_path, 'path_to_archive_folder')  # Use this to move the file