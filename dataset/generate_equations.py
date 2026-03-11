# dataset/generate_equations.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : Generate synthetic handwritten equations
# -------------------------------------------------


# =============================================
# Importing the Necessary Libraries
# ----------------------------------------
# os for handling file paths and directories.
# random for generating random equations and variations.
# numpy for numerical operations.
# PIL (Pillow) for image creation and manipulation.
# =============================================
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Output folder for generated images and labels
OUTPUT_FOLDER = "dataset/generated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Image dimensions and dataset size
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
DATASET_SIZE = 50000

# Digits and operators for equation generation
digits = "0123456789"
plus_minus = ["+", "-"]
multiply_symbols = ["*", "×", "·"]
divide_symbols = ["/", "÷"]
operators = plus_minus + multiply_symbols + divide_symbols

# Fonts for rendering equations
FONTS = [
    "dataset/fonts/PatrickHand-Regular.ttf",
    "dataset/fonts/IndieFlower-Regular.ttf",
    "dataset/fonts/Caveat-VariableFont_wght.ttf"
]

# =============================================
# generate_equation_function to create random mathematical equations.
# ----------------------------------------
# The function randomly determines the length of the equation (between 2 to 4 operators) and constructs the equation by randomly selecting digits and operators.
# It also randomly adds parentheses to increase the complexity of the equations. 
# Finally, it appends an equals sign at the end of the equation to complete it. 
# The generated equation is returned as a string.
# =============================================
def generate_equation():
    # Randomly determine the length of the equation (number of operators)
    length = random.randint(2,4)
    eq = ""
    for i in range(length):
        eq += random.choice(digits)
        if i < length-1:
            op = random.choice(operators)
            eq += op

    # Add parentheses randomly
    if random.random() > 0.6 and len(eq) > 3:
        split_index = random.randint(1, len(eq)-2)
        eq = "(" + eq[:split_index] + ")" + eq[split_index:]

    # Add equals symbol
    eq += "="

    return eq

# =============================================
# generate_image function to create an image of the generated equation with random fonts, sizes, rotations, and noise.
# ----------------------------------------
# The function creates a blank white image and uses the PIL library to draw the generated equation onto the image using a randomly selected font and size.
# It then applies random rotations, blurring, and noise to the image to make it more realistic.
# Finally, the image is saved to the specified output folder with a unique filename, and the path to the saved image is returned.
# =============================================
def generate_image(text, index):
    # Create a blank white image
    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
   
    # Draw the equation text on the image
    draw = ImageDraw.Draw(img)
    font_path = random.choice(FONTS)
    font_size = random.randint(32,42)
    font = ImageFont.truetype(font_path, font_size)
    x = random.randint(20,60)
    y = random.randint(10,30)
    draw.text((x,y), text, font=font, fill=0)

    # Random rotation
    angle = random.randint(-7,7)
    img = img.rotate(angle, fillcolor=255)

    # Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1)))

    # Noise
    img_array = np.array(img)
    noise = np.random.normal(0,8,img_array.shape)
    img_array = img_array + noise
    img_array = np.clip(img_array,0,255)

    # Convert back to PIL image
    img = Image.fromarray(img_array.astype(np.uint8))

    # Save the image
    path = f"{OUTPUT_FOLDER}/eq_{index}.png"
    img.save(path)

    return path

# =============================================
# main function to generate the dataset of equations and corresponding images.
# ----------------------------------------
# The function iterates for a specified number of times (DATASET_SIZE) to generate random equations and their corresponding images.
# It calls the generate_equation function to create a random equation and then uses the generate_image function to create an image of that equation.
# The path to the saved image and the corresponding equation are stored in a list, which is then written to a CSV file in the output folder for later use in training the model.
# =============================================
def main():
    # List to store the paths and equations for the dataset
    labels = []

    # Generate equations and corresponding images
    for i in range(DATASET_SIZE):
        eq = generate_equation()
        path = generate_image(eq, i)
        labels.append(f"{path},{eq}")
        if i % 5000 == 0:
            print("Generated", i)
    with open(f"{OUTPUT_FOLDER}/labels.csv", "w") as f:
        for line in labels:
            f.write(line + "\n")

# Entry point of the script
if __name__ == "__main__":
    main()