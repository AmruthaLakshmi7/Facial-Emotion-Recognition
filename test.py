import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("emotion_model.h5")


emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


image_path = "test/sad/PrivateTest_552501.jpg"   


img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (48, 48))
gray = gray / 255.0
gray = np.reshape(gray, (1, 48, 48, 1))


prediction = model.predict(gray)[0]
emotion_index = np.argmax(prediction)
emotion = emotion_labels[emotion_index]
accuracy = prediction[emotion_index] * 100

print("Predicted Emotion:", emotion)
print("Accuracy:", f"{accuracy:.2f}%")

