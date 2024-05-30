import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path of the Image
data_directory = ' Path of the file '

# classes in your dataset
categories = ['Ancient Egyptian architecture', 'Art Nouveau architecture']

# List to store image data and labels
image_data = []

# Pre-processing
for category in categories:
    category_path = os.path.join(data_directory, category)
    label = categories.index(category)
    
    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)
        img = cv2.imread(img_path, 0)  # Read the image in grayscale
        
        try:
            img = cv2.resize(img, (50, 50))
            
            # Local Binary Pattern (LBP)
            lbp_image = local_binary_pattern(img, P=64, R=8, method='uniform')
            # Using LBP image as features
            features = lbp_image.flatten()
            image_data.append([features, label])
        except Exception as e:
            print(f"Error processing image: {img_path}, {str(e)}")

features, labels = zip(*image_data)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
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
