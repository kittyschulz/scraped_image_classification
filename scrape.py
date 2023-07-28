import os
import random
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from PIL import Image

from typing import List

def scrape_google_images(label: str, num_images: int, verbose: bool = False) -> None:
    """
    Scrapes Google Images for a given label and downloads a specified number of images.

    Args:
        label (str): The search term for images.
        num_images (int): The number of images to download.
        verbose (bool, optional): Whether to print download status. Defaults to False.

    Returns:
        None, downloads images to local directory.

    Raises:
        None
    """

    # Replace spaces in the label with '+'
    query = quote_plus(label)
    url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"

    # Send a request to Google Images
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')

    # Create a directory with the name of the label
    folder_name = label.replace(' ', '_')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Extract image URLs from the page
    image_urls = []
    for img in soup.find_all('img'):
        img_url = img.get('src')
        if img_url and img_url.startswith('https://'):
            image_urls.append(img_url)

    # Download and save the images locally
    for i, img_url in enumerate(image_urls[:num_images]):
        try:
            response = requests.get(img_url)
            img_name = f"{folder_name}/{folder_name}_{i + 1}.jpg"
            with open(img_name, "wb") as f:
                f.write(response.content)
            if verbose:
                print(f"Downloaded {img_name}")
        except:
            print(f"Failed to download {img_url}")


def visualize_sample_images(classes: List[str], num_samples_per_folder: int = 1) -> None:
    """
    Visualizes a sample of images from each class. 

    Args:
        classes (List[str]): A list of class names (folder names containing images).
        num_samples_per_folder (int, optional): The number of images to display per class.
            Defaults to 1.

    Returns:
        None, visualizes the images.

    Raises:
        None
    """

    folder_names = [i.replace(' ', '_') for i in classes]
    fig, axs = plt.subplots(1, len(folder_names), figsize=(15, 5))

    for i, folder_name in enumerate(folder_names):
        # Get a list of image filenames in the folder
        image_files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]

        # Randomly select a sample of images
        sampled_images = random.sample(image_files, min(num_samples_per_folder, len(image_files)))

        # Display the images in the Jupyter Notebook
        for img_file in sampled_images:
            img_path = os.path.join(folder_name, img_file)
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].axis('off')

    plt.show()


def main():
    labels = input("Enter the labels of the type of images to scrape, comma separated (e.g., horse,dog,cat): ")
    num_images = int(input("Enter the number of images to scrape and download for each label: "))

    labels = labels.split(",")
    for label in labels:
        scrape_google_images(label, num_images)


if __name__ == "__main__":
    main()