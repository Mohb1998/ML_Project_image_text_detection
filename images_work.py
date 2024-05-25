from PIL import Image
import os

# Directory containing images
image_dir = "./input/test-letters"
#directory_path = "./input/handwritten-characters/Train"
directory_path = "./input/handwritten-characters/Validation"

for label in os.listdir(directory_path):
    if label in ["#", "$", "&", "@"]:
        continue
    image_dir = os.path.join(directory_path, label)
    # Iterate through all files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open an image file
            img_path = os.path.join(image_dir, filename)
            with Image.open(img_path) as img:
                # Reverse the colors
                inverted_image = Image.eval(img, lambda x: 255 - x)
                # Save the modified image back to the directory
                inverted_image.save(img_path)
    print("folder ", label, "finished.")

print("Color inversion completed.")



