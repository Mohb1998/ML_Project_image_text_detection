from PIL import Image
import os

# INVERSION OF COLORS OF IMAGES in a folder

#image_dir = "./input/test-letters"
#directory_path = "./input/handwritten-characters/Train"
directory_path = "./input/handwritten-characters/Validation"

for label in os.listdir(directory_path):
    if label in ["#", "$", "&", "@"]:
        continue
    image_dir = os.path.join(directory_path, label)
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(image_dir, filename)
            with Image.open(img_path) as img:
                inverted_image = Image.eval(img, lambda x: 255 - x)
                inverted_image.save(img_path)
    print("folder ", label, "finished.")

print("Color inversion completed.")



