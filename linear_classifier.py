'''Linear classifier(SVM loss function,L2 regularization,gradient descent weight matrix optimization)'''
'''Dataset credit: SHREYA.MAHER (kaggle)
    Dataset name: Fruits Dataset (Images)'''

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_dataset(dataset_path, size=(64, 64), normalize=False):
    image_data = []
    labels = []
    class_names = []
    
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
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, size)
            img_vector = img.flatten()
            if normalize:
                img_vector = img_vector / 255.0
            image_data.append(img_vector)
            labels.append(class_names.index(foldername))
    return np.array(image_data), np.array(labels), class_names

def svm_loss(W, train_images, train_labels, margin=1.0, reg_strength=0.1):
    num_samples = train_images.shape[0] #train images shape is (num_samples, num_features) and shape of weights is (num_classes, num_features)
    scores = np.dot(train_images, W.T) #shape of scores is (num_samples, num_classes)
    correct_class_scores = scores[np.arange(num_samples), train_labels].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + margin) #shape of margins is (num_samples, num_classes), means how much the score of the correct class exceeds the margin
    margins[np.arange(num_samples), train_labels] = 0 #because we don't want to consider the correct class in loss calculation
    loss = np.sum(margins) / num_samples
    
    loss += 0.5 * reg_strength * np.sum(W * W) # with L2 regularization
    #we calculate the gradient
    binary = margins > 0 #means how many many samples have contributed to the loss,shape if binary is (num_samples, num_classes)
    row_sum = np.sum(binary, axis=1) # Count how many classes exceeded the margin, axis 1 means row-wise sum
    binary[np.arange(num_samples), train_labels] = -row_sum #adjusts the correct class elements to account for the number of incorrect classes that exceeded the margin.
    gradient = np.dot(binary.T, train_images) / num_samples
    
    # gradient of loss with respect to weights
    gradient += reg_strength * W
    
    return loss, gradient

    '''Positive values in binary → Positive gradient → Subtract during update → Weights decrease → Wrong class scores decrease
    Negative values in binary → Negative gradient → Subtract negative during update → Weights increase → Correct class score increases
    '''
class Train_predict_accuracy:
    @staticmethod
    def train_svm(train_images, train_labels, num_classes, num_features, learning_rate=1e-3, epochs=1000, reg_strength=0.1,tolerance=1e-5):
        W = np.random.randn(num_classes, num_features) * 0.01  # Initialize weights
        for epoch in range(epochs):
            loss, grad = svm_loss(W, train_images, train_labels, reg_strength=reg_strength)
            diff=W - learning_rate * grad
            if np.all(np.abs(diff) < tolerance):
                break
            # if epoch % 10 == 0: #after every 10 epochs, print the loss
            #     print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
            
        return diff

    @staticmethod
    def predict(W, images):
        scores = np.dot(images, W.T)
        return np.argmax(scores, axis=1)

if __name__ == '__main__':
    dataset_folder = "Fruits"
    images, labels, class_names = load_dataset(dataset_folder, size=(64, 64), normalize=True)
    num_classes = len(class_names)
    num_features = images.shape[1]

    # Split the dataset into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    TPA = Train_predict_accuracy()
    W = TPA.train_svm(train_images, train_labels, num_classes, num_features)
   
    # Predict on the test set
    test_predictions = TPA.predict(W, test_images)

    # Calculate accuracy
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test set accuracy: {test_accuracy*100}")