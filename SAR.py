import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

data_dir = "C:\\Users\\Faraz\\Desktop\\flask"
# categories = ["FIGHTING", "WRITING", "SITTING", "USE_MOBILE", "SLEEPING"]
categories = ["FIGHTING", "SITTING", "SLEEPING", "USE_MOBILE", "WRITING"]
# Function to create a simple model
def load_data(data_dir):
    images = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (100, 100))
            image = image / 255.0

            images.append(image)
            labels.append(class_num)

    return np.array(images), np.array(labels)

# Load and split the dataset
X, y = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def create_model(epochs=10):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model
# def load_model_weights(model, weights_path):
#     model.load_weights(weights_path)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    return image
# latest_model_filename = "activity_model_updated.h5"
# model = create_model()
# load_model_weights(model, latest_model_filename)
# model.save(f"{latest_model_filename[:-6]}_updated.keras")
# label_encoder = LabelEncoder()
# labels = np.array(categories)
# label_encoder.fit_transform(labels)
# np.savetxt('label.txt', label_encoder.classes_, fmt='%s')
# print(f"Categories in SAR.py: {label_encoder.classes_}")


