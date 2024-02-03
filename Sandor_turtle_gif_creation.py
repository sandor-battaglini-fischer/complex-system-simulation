import turtle
import math
import random
import os
from PIL import Image
import imageio


images = []
for i in range(1804):
    try:
        img = Image.open(f"frame_{i:04}.png")
        images.append(img)
    except IOError:
        print(f"Could not read file frame_{i:04}.png")

if images:
    imageio.mimsave('heart_animation.gif', images, duration=10)
    print("GIF animation created.")
else:
    print("No images were loaded. GIF creation skipped.")