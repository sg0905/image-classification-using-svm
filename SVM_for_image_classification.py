import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Directory where the image data is stored
data_directory = 'D:\\SVM\\data'

# Categories or classes in your dataset
categories = ['Ancient Egyptian architecture', 'Art Nouveau architecture']

# List to store image data and labels
image_data = []

# Load and preprocess the images
for category in categories:
    # Create the path to the directory for the current category
    category_path = os.path.join(data_directory, category)
    
    # Assign a label (numeric identifier) to the category
    label = categories.index(category)

    # Loop through each image in the category
    for img_file in os.listdir(category_path):
        # Construct the full path to the image
        img_path = os.path.join(category_path, img_file)
        
        # Read the image in grayscale
        img = cv2.imread(img_path, 0)
        
        try:
            # Resize the image to 50x50 pixels
            img = cv2.resize(img, (50, 50))
            
            # Apply Local Binary Pattern (LBP)
            lbp_image = local_binary_pattern(img, P=64, R=8, method='uniform')
            
            # Use LBP image directly as features
            features = lbp_image.flatten()
            
            # Append the features and its label to the data list
            image_data.append([features, label])
        except Exception as e:
            # Handle errors that may occur during image processing
            print(f"Error processing image: {img_path}, {str(e)}")

# Split data into features (X) and labels (y)
features, labels = zip(*image_data)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Train an SVM classifier with a linear kernel
svm_model = SVC(kernel='linear')  # You can experiment with different kernels

# Fit the SVM model on the training data
svm_model.fit(X_train, y_train)

# Evaluate the SVM on the testing data
y_pred = svm_model.predict(X_test)

# Calculate and print the accuracy of the SVM model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can also visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = categories
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
