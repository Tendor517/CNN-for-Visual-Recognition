'''To use L1 distance metrics '''
'''Dataset credit: SHREYA.MAHER (kaggle)
    Dataset name: Fruits Dataset (Images)'''

# import the modules
import os
from os import listdir
import cv2
import numpy as np


#arrays used
image_data = []
labels = []
class_names = []

#l1 distance formula
def l1_distance(img_vec1,img_vec2):
    return np.sum(abs(img_vec1-img_vec2))

# preprocess images
def preprocess_image(image_path, size=(64, 64), normalize=False):
    """
    Preprocess an image for L1 distance-based classification.
    Args:
        image_path (str): Path to the image.
        size (tuple): Desired size (width, height) for resizing.
        normalize (bool): Whether to normalize pixel values to [0, 1].
    Returns:
        np.ndarray: Preprocessed 1D image array.
    """
    # Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found or cannot be read.")
    
    # Resize the image
    img = cv2.resize(img, size)
    
    # Flatten the image to 1D
    img_vector = img.flatten()
    
    # Normalize pixel values if required
    if normalize:
        img_vector = img_vector / 255.0

    return img_vector

def load_dataset(dataset_path, size=(64, 64), normalize=False):
    
    for foldername in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, foldername)
        if not os.path.isdir(folder_path):  # Skip non-directory files
            continue
        class_names.append(foldername)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Skipping these unsupported file format: {filename}")
                continue
            img_path = os.path.join(folder_path, filename)
            image_data.append(preprocess_image(img_path, size=size, normalize=normalize))
            labels.append(foldername)
    return np.array(image_data), np.array(labels), class_names

#l1 distance classifier
class L1DistanceClassifier:
    def __inti__(self):
        self.train_images=None
        self.train_labels=None

    def train(self,images,labels): #assigning/just storing image and labels to the classifier object
        self.train_images=images
        self.train_labels=labels

    def predict(self,image):
        min_distance=float('inf')
        min_label=None
        for i in range(len(self.train_images)):
            distance=l1_distance(image,self.train_images[i])
            if distance<min_distance:
                min_distance=distance
                min_label=self.train_labels[i]
        return min_label

        

#main function
if __name__ == '__main__':
    #laod data and extract images, labels and classes
    dataset_folder = "Fruits"
    images, labels, class_names = load_dataset(dataset_folder, size=(64, 64), normalize=True)

    #training stage
    L1_classifier=L1DistanceClassifier()
    L1_classifier.train(images,labels)

    #testing stage
    test_image_path=dataset_folder+"/"+"apple/image_1.jpg"
    test_image=preprocess_image(test_image_path,size=(64,64),normalize=True)
    predict_label=L1_classifier.predict(test_image)
    print(f"Predicted Label of {test_image}= {predict_label}")