import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from keras_cv_attention_models import davit

from typing import List, Union, Optional


def load_image(img_path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Loads an image from the provided path.

    Args:
        img_path (Union[str, os.PathLike]): The file path of the image to be loaded.

    Returns:
        np.ndarray: A NumPy array representing the image.

    Raises:
        None
    """

    img = image.load_img(img_path)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_scraped_images(object_classes: List[str],
                           output_file: str = 'classification_results.xlsx',
                           return_df: bool = True) -> Optional[pd.DataFrame]:
    """
    Predicts the class of scraped images using tiny DaViT and saves the results in an Excel file.

    Keras DaViT implementation pretrained on Imagenet: 
    https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/davit

    Args:
        object_classes (List[str]): A list of object class names (folder names containing images).
        output_file (str, optional): The name of the Excel file to save the classification results.
            Defaults to 'classification_results.xlsx'.
        return_df (bool, optional): Whether to return the classification results as a DataFrame.
            Defaults to True.

    Returns:
        pd.DataFrame (optiona1): A DataFrame containing the classification results if return_df is True,
            otherwise None. The DataFrame is also saved locally as an *.xlsx file.

    Raises:
        None
    """
    # Load DaViT
    model = davit.DaViT_T(pretrained="imagenet")

    # Get folder names for object classes
    folder_names = [i.replace(' ', '_').lower() for i in object_classes]

    # Create lists to store results
    image_names = []
    classifications = []
    confidence = []

    for folder_name in folder_names:
        for img_file in os.listdir(folder_name):
            img_path = os.path.join(folder_name, img_file)
            img = load_image(img_path)
            preds = model(model.preprocess_input(img))
            decoded_preds = model.decode_predictions(preds)[0]

            image_names.append(img_file)
            classifications.append(decoded_preds[0][1].replace('_', ' ').title())
            confidence.append(decoded_preds[0][2])

    # Create a DataFrame
    results_df = pd.DataFrame({
        'Image Name': image_names,
        'Classification': classifications,
        'Confidence': confidence
    })

    # Save the DataFrame to an Excel file
    results_df.to_excel(output_file, index=False)
    print(f"Classification results saved to {output_file}")

    if return_df:
        return results_df
    

def find_directories_with_images():
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'] 

    image_directories = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_directories.append(root.replace('./',''))
                break  # Break to avoid adding the same directory multiple times

    return image_directories


def main():
    object_classes = input("Enter the labels of the type of images to run inference on, comma separated. No input runs inference on all images in current directory : ")
    object_classes = object_classes.split(",")
    if object_classes == ['']:
        object_classes = find_directories_with_images()
        print(f'Running inference on the following classes: {[obj.replace("_"," ").title() for obj in object_classes]}')
    predict_scraped_images(object_classes, return_df=False)


if __name__ == "__main__":
    main()