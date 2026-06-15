import os

IMG_DIR = "data/images"

print("Images found:")
for f in os.listdir(IMG_DIR):
    print(f)
