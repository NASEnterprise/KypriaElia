import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(data_file):
    data = np.load(data_file)
    return data['image_data'], data['labels']

def train_model(image_data, labels):
    # Check the number of samples
    print(f"Number of samples: {image_data.shape[0]}")
    
    # Split the data into training and testing sets
    if image_data.shape[0] > 1:
        X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
    else:
        raise ValueError("Not enough samples to split into training and testing sets.")

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear')

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    return svm_classifier

def save_model(model, model_file):
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    data_file = "preprocessed_data.npz"
    model_file = "svm_model.joblib"

    # Load the preprocessed data
    image_data, labels = load_data(data_file)

    # Train the SVM model
    model = train_model(image_data, labels)

    # Save the trained model
    save_model(model, model_file)