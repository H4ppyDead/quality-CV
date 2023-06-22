import os
import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def label_from_filename(filename):
    if "cast_ok" in filename:
        return 1
    elif "cast_def" in filename or "new__" in filename:
        return 0

def load_images_from_folder(folder):
    images = []
    labels = []
    
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(subdir, file)  # Define file_path here
            label = label_from_filename(file)
            if label is None:  # Skip this file if it doesn't have a valid label
                print(f"Skipping file {file_path} with label {label}")
                continue
            img = cv2.imread(file_path)
            if img is None:
                print(f"Cannot read file {file_path}.")
                continue
            img = cv2.resize(img, (64, 64))  # Resize image
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label)  # Assign label based on the filename
    return np.array(images), np.array(labels)



train_images_path = r"C:\Users\yanto\Desktop\quality control\casting_data\train"
test_images_path = r"C:\Users\yanto\Desktop\quality control\casting_data\test"


# Load images
X_train, y_train = load_images_from_folder(train_images_path)
X_test, y_test = load_images_from_folder(test_images_path)

print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)

# Initialize model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Fit the model
model.fit(X_train, y_train, epochs=10)

# Test the model
model.evaluate(X_test, y_test)

def classify_defect(probability, thresholds=[0.33, 0.67]):
    if probability < thresholds[0]:
        return 'No/Minor defect'
    elif probability < thresholds[1]:
        return 'Defect, but fixable'
    else:
        return 'Defect, unfixable'

# prediction = model.predict(some_image)  # Получаем вероятность
# category = classify_defect(prediction[0])  # Классифицируем вероятность
# print(category)
