'''To implment K-fold Cross Validation, with hyper parameter tuning considering the best K and distance metric (l1 or l2)
    & trying random random_state values for splitting data and at kFold initialization'''
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

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import random



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

# Main function
if __name__ == '__main__':
    # Load data and extract images, labels, and classes
    dataset_folder = "Fruits"
    images, labels, class_names = load_dataset(dataset_folder, size=(64, 64), normalize=True)

    # Hyperparameter tuning and K-Fold with multiple seeds
    num_experiments = 5
    test_accuracies = []
    


    for seed in random.sample(range(1, 123), num_experiments):  # Pick 5 random seeds
        print(f"\nExperiment with random seed {seed}")
        
        # Split dataset using the seed
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            images, labels, test_size=0.4, random_state=seed
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.1, random_state=seed
        )
        # Initialize K-Fold
        kf = KFold(n_splits=5, shuffle=True,random_state=seed)
        
        # Hyperparameters
        best_k = None
        best_metric = None
        best_accuracy = 0

        # K-Fold Cross-Validation
        for train_indices, val_indices in kf.split(train_images):
            fold_train_images, fold_val_images = train_images[train_indices], train_images[val_indices]
            fold_train_labels, fold_val_labels = train_labels[train_indices], train_labels[val_indices]

            # Initialize your classifier
            classifier = DistanceClassifier()
            classifier.train(fold_train_images, fold_train_labels)

            # Hyperparameter tuning
            for k in [2, 3, 6, 7, 10]:
                for metric in ['l1', 'l2']:
                    correct_predicted = 0
                    for i in range(len(fold_val_images)):
                        predicted_label = classifier.predict(fold_val_images[i], k, metric)
                        if predicted_label == fold_val_labels[i]:
                            correct_predicted += 1
                    accuracy = correct_predicted / len(fold_val_images)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_k = k
                        best_metric = metric

        print(f"Best K: {best_k}, Best Metric: {best_metric}, Best Accuracy (Validation): {best_accuracy:.4f}")

        # Final testing on test set
        classifier = DistanceClassifier()
        classifier.train(train_images, train_labels)  # Train on the entire training set
        correct_predictions = 0
        for i in range(len(test_images)):
            predicted_label = classifier.predict(test_images[i], k=best_k, metric=best_metric)
            if predicted_label == test_labels[i]:
                correct_predictions += 1
        test_accuracy = correct_predictions / len(test_images)
        print(f"Accuracy on test set: {test_accuracy:.4f}")
        test_accuracies.append(test_accuracy)

    # Report mean and standard deviation of test accuracy
    print("\nFinal Results:")
    print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(test_accuracies):.4f}")

        


