import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset (images and labels)
def load_dataset(dataset_path):
    X, y = [], []
    label_map, label_reverse_map = {}, {}
    label_index = 0

    for label_name in os.listdir(dataset_path):
        label_map[label_name] = label_index
        label_reverse_map[label_index] = label_name
        label_index += 1

        label_folder = os.path.join(dataset_path, label_name)
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            X.append(img_path)
            y.append(label_map[label_name])

    return X, y, label_map, label_reverse_map

# Preprocess the image (convert to RGB)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Extract features using DeepFace
def extract_features(X, y):
    features, updated_labels = [], []
    for i, img_path in enumerate(X):
        try:
            img = preprocess_image(img_path)
            embedding = DeepFace.represent(img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            features.append(embedding)
            updated_labels.append(y[i])
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
    return np.array(features), np.array(updated_labels)

# Webcam live face recognition function with face detection
def live_face_recognition(model, label_reverse_map):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Extract the face region of interest (ROI)
            face_roi = rgb_frame[y:y+h, x:x+w]

            try:
                # Extract features using DeepFace
                embedding = DeepFace.represent(face_roi, model_name='VGG-Face', enforce_detection=False)[0]['embedding']

                # Predict the identity using the trained model
                prediction = model.predict([embedding])
                label = label_reverse_map.get(prediction[0], "Unknown")

                # Draw a bounding box around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error detecting face: {e}")
                cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow("Webcam - Face Detection & Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main Function to Train Model and Test on Webcam
def main():
    dataset_path = 'data1'  # Set your dataset path here

    # Load and preprocess the dataset
    X, y, label_map, label_reverse_map = load_dataset(dataset_path)
    features, updated_labels = extract_features(X, y)

    # Check if features were extracted successfully
    if len(features) == 0:
        print("No features extracted. Check your dataset.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, updated_labels, test_size=0.2, random_state=42)

    # Train the SVM classifier
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Start live face recognition using the webcam
    live_face_recognition(svm_model, label_reverse_map)

# Run the main function
if __name__ == '__main__':
    main()
