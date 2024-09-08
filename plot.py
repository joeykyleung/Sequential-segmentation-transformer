import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def extract_loss_from_log(log_file_path):
    epoch_loss = []

    # Regular expression to match the relevant parts of the log line
    pattern = re.compile(r'Epoch: (\d+) / \d+ \| loss for slice \d+: ([\d.]+)')
    
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_loss.append((epoch, loss))
    
    return epoch_loss

def plot_losses(log_files, degree=3):
    plt.figure(figsize=(12, 8))

    for log_file in log_files:
        epoch_loss = extract_loss_from_log(log_file)
        epochs, losses = zip(*epoch_loss)
        
        # Fit a polynomial to the data
        coeffs = np.polyfit(epochs, losses, degree)
        polynomial = np.poly1d(coeffs)
        fit_epochs = np.linspace(min(epochs), max(epochs), 100)
        fit_losses = polynomial(fit_epochs)
        
        # plt.plot(epochs, losses, 'o', label=f'Raw data {log_file}')
        plt.plot(fit_epochs, fit_losses, '-', label=f'Fit curve {log_file}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs with Curve of Best Fit for Multiple Files')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    
def change_white_to_black(image_path):
    # Load the image
    img = Image.open(image_path)

    # Convert the image to RGB mode if it's not
    img = img.convert('RGB')

    # Get the pixel data
    pixels = img.load()

    # Get image dimensions
    width, height = img.size

    # Loop through all pixels and change white pixels to black
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # if r == 255 and g == 255 and b == 255:  # Check if the pixel is white
            if (r + g + b )/ 3 > 245:
                pixels[x, y] = (0, 0, 0)  # Change to black

    # Save the modified image with a new name
    output_path = image_path.replace('.png', '_black.png')
    img.save(output_path)
    print(f"Processed {image_path} and saved as {output_path}.")


def process_images(image_paths):
    for image_path in image_paths:
        change_white_to_black(image_path)
    

def main():
    log_files = [
        'logs/train_SST - best.log',
        'logs/train_SST_test - 25.log',
        'logs/train_SST_test - 50.log'
    ]
    # plot_losses(log_files)
    image_paths = [
        'results/SST/output_2000.png',
    ]
    process_images(image_paths)

if __name__ == '__main__':
    main()
