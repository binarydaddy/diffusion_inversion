import glob
import random
import os
import shutil

p = '/data/inversion_data/0712_image_data'

dst = '/data/inversion_data/validation_data'

plist = sorted(glob.glob(os.path.join(p, "*.pt")))
namelist = [os.path.basename(p).split("_")[0] for p in plist]

# Sample 100 items with replacement
random_sample_names = random.choices(namelist, k=500)

for names in random_sample_names:
    img_path = f"{p}/{names}_final.png"
    metadata_path = f"{p}/{names}_metadata.json"
    data_path = f"{p}/{names}_data.pt"

    shutil.copy(img_path, os.path.join(dst, f"{names}_final.png"))
    shutil.copy(metadata_path, os.path.join(dst, f"{names}_metadata.json"))
    shutil.copy(data_path, os.path.join(dst, f"{names}_data.pt"))