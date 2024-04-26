import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Directory where the image has been stored
dir = 'D:\\SVM\\data'

# Categories used
categories = ['American craftsman style', 'Russian Revival architecture']

# List to store image data and labels
data = []

# Load and preprocess the images
for category in categories:
    # Create the path to the directory for the current category
    path = os.path.join(dir, category)
    
    # Assign a label (numeric identifier) to the category
    label = categories.index(category)

    # Loop through each image in the category
    for img in os.listdir(path):
        # Construct the full path to the image
        imgpath = os.path.join(path, img)
        
        # Read the image in grayscale
        a_img = cv2.imread(imgpath, 0)
        
        try:
            # Resize the image to 50x50 pixels
            a_img = cv2.resize(a_img, (50, 50))
            
            # Apply Local Binary Pattern (LBP)
            lbp_image = local_binary_pattern(a_img, P=8, R=1, method='uniform')
            ulbp_image = local_binary_pattern(a_img, P=16, R=2, method='uniform')
            
            # Concatenate LBP and ULBP histograms with the flattened image
            features = np.concatenate([np.array(a_img).flatten(), lbp_image.flatten(), ulbp_image.flatten()])
            
            # Append the features and its label to the data list
            data.append([features, label])
        except Exception as e:
            # Handle errors that may occur during image processing
            print(f"Error processing image: {imgpath}, {str(e)}")

# Split data into features (X) and labels (y)
features, labels = zip(*data)

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

# Construct the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
