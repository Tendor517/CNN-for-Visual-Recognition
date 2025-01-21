'''Softmax Classifier with Gradient Descent and L2 Regularization'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from linear_classifier import load_dataset


class SoftmaxClassifier:
    def __init__(self):
        self.W = None

    def softmax(self, scores):
        """
        Compute the softmax probabilities for each class. 
        Parameters: score shape is (num_samples, num_classes)
        Returns: numpy.ndarray: The softmax probabilities for each class.
        """
        # Subtract the maximum score in each row from each score in that row for numerical stability
        # This prevents large exponentials which can cause overflow
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        # Divide by the sum of the exponentiated scores to get probabilities
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy_loss(self, probs, labels):
        """
        Compute the cross-entropy loss.
        """
        num_samples = labels.shape[0] #shape of labels is (num_samples,)
        correct_logprobs = -np.log(probs[np.arange(num_samples), labels]) #correct_logprobs shape is (num_samples,)
        loss = np.sum(correct_logprobs) / num_samples
        return loss

    def compute_gradient(self, probs, train_images, train_labels):
        """
        Compute the gradient of the loss with respect to weights.
        """
        num_samples = train_images.shape[0]
        dscores = probs
        dscores[np.arange(num_samples), train_labels] -= 1  # Adjust probabilities for correct labels, shape is (num_samples, num_classes)
        dscores /= num_samples
        '''Issues without averaging:
            Weights grow exponentially with batch size
            Learning rate needs constant adjustment
            Model may fail to converge
            Potential numerical overflow
            Unstable training process'''
        gradient = np.dot(dscores.T, train_images)  # Gradient with respect to weights
        return gradient

    def train(self, train_images, train_labels, num_classes, learning_rate=0.01, epochs=100, reg_strength=0.1):
        """
        Train the softmax classifier using gradient descent.
        """
        num_samples, num_features = train_images.shape
        self.W = np.random.randn(num_classes, num_features) * 0.01  # Initialize weights

        for epoch in range(epochs):
            #class scores
            scores = np.dot(train_images, self.W.T)

            # Calculate softmax probabilities
            probs = self.softmax(scores)

            # Compute loss with L2 regularization
            loss = self.cross_entropy_loss(probs, train_labels)
            loss += 0.5 * reg_strength * np.sum(self.W * self.W)  # Add L2 regularization

            # Calculate gradients
            gradient = self.compute_gradient(probs, train_images, train_labels)
            gradient += reg_strength * self.W  

            # Update weights
            self.W -= learning_rate * gradient

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, images):
        """
        Predict class labels for a set of images.
        """
        scores = np.dot(images, self.W.T)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)


# Main function
if __name__ == '__main__':
    dataset_folder = 'Fruits'
    images, labels, class_names = load_dataset(dataset_folder, size=(64, 64), normalize=True)
    num_classes = len(class_names)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Initialize and train the softmax classifier
    softmax_classifier = SoftmaxClassifier()
    softmax_classifier.train(train_images, train_labels, num_classes)

    # Prediction on the test set
    test_predictions = softmax_classifier.predict(test_images)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test set accuracy: {test_accuracy:.4f}")
