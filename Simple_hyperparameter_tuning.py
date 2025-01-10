'''To implment Hyperparameter tuning on KNN algorithm to find the best value for K and Distance metric (in our case either l1 or l2)'''
'''Dataset credit: SHREYA.MAHER (kaggle)
    Dataset name: Fruits Dataset (Images)'''

# import the modules
import os
from os import listdir
import cv2
import numpy as np
from  collections import Counter
from itertools import islice
from sklearn.model_selection import train_test_split



#l1 or l2 distance formula
def either_distance(img_vec1,img_vec2,metric='l1'):
    if metric=='l1':
        return np.sum(abs(img_vec1-img_vec2))
    elif metric=='l2':
        return np.sqrt(np.sum((img_vec1-img_vec2)**2))
    else:
        raise ValueError(f"distance metric: {metric} not included yet for training")

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

#Load Dataset
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


class DistanceClassifier:
    def __init__(self):
        self.train_images = None
        self.train_labels = None

    def train(self, images, labels):
        self.train_images = images
        self.train_labels = labels

    def predict(self, image, k=3,metric='l1'):
        distances = []
        for i in range(len(self.train_images)):
            distance = either_distance(image, self.train_images[i],metric='l1')  #l1 distance as default
            distances.append((self.train_labels[i],distance)) #list of tuples

        # Sort distances and select the nearest k neighbors
        distances.sort(key=lambda x: x[1])
        k_nearest_neighbors = distances[:k]

        # Determine the most common label among the k nearest neighbors
        labels = [label for label, _ in k_nearest_neighbors] #list of labels
        most_common_label = Counter(labels).most_common(1)[0][0]
        return most_common_label

#arrays used
image_data = []
labels = []
class_names = []


# main function
if __name__ == '__main__':
    # Load data and extract images, labels, and classes
    dataset_folder = "Fruits"
    images, labels, class_names = load_dataset(dataset_folder, size=(64, 64), normalize=True)
    # print(images.shape)
    # print(labels.shape)
    # print(class_names)
    # Training stage
    Distance_classifier = DistanceClassifier()
    Distance_classifier.train(images, labels)

    # Testing on a single image
    test_image_path = dataset_folder + "/" + "apple/image_1.jpg"
    test_image = preprocess_image(test_image_path, size=(64, 64), normalize=True)
    predict_label = Distance_classifier.predict(test_image,k=5)
    print(f"Predicted Label of {test_image_path} image = {predict_label}")

    
    #SPLIT DATASET INTO TRAIN, VALIDATION AND TEST SETS (to perform hyperparameter tuning)
     
    # Split into train and temp (temp will be split into validation and test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.4, random_state=42  #40% of data goes to temp set
    )

    # splitting temp into validation and test
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42  #50% of data goes to test, so 50/50 here
    )

    '''
        print(f"Training Set: {len(train_images)} samples")
        print(f"Validation Set: {len(val_images)} samples")
        print(f"Test Set: {len(test_images)} samples")
    '''

    #hyperparameters
    best_k=None
    best_metric=None
    best_accuracy=0

    #IMPLEMENNT HYPERPARAMETER TUNING, by trying different k values with l1 and l2
    for k in [1,3,5,7,9]:
        for metric in ['l1','l2']:
            correct_predicted_labels=0
            for i in range(len(val_images)):
                predict_label=Distance_classifier.predict(val_images[i],k,metric)
                if predict_label==val_labels[i]:
                    correct_predicted_labels+=1
            accuracy=correct_predicted_labels/len(val_images)
            print(f"for K={k} and metric= {metric}, Validation accuracy= {accuracy:.4f}")
            if accuracy>best_accuracy:
                best_accuracy=accuracy
                best_k=k
                best_metric=metric
    #best k and metric based on accuracy on validation set
    print(f"Therefore, best hyperparameters K= {best_k} and best metric={best_metric} with accuaracy of ={accuracy:.4f} or {accuracy*100}%")

    #evaluating final performance on test image set, once the best K and distance metric is found
    correct_predictions=0
    for i in range(len(test_images)):
        predicted_label=Distance_classifier.predict(test_image[i],k=best_k,metric=best_metric)
        if predict_label==test_labels[i]:
            correct_predictions+=1
    test_accuracy=correct_predictions/len(test_images)
    print(f"Final performance on Test set, test set accuracy= {test_accuracy:.4f} or {test_accuracy*100}%")