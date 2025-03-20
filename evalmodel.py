import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def load_data(data_file):
    data = np.load(data_file)
    return data['image_data'], data['labels']

def evaluate_model(model_file, data_file):
    # Load the preprocessed data
    image_data, labels = load_data(data_file)

    # Load the trained model
    model = joblib.load(model_file)

    # Make predictions on the entire dataset
    y_pred = model.predict(image_data)

    # Generate the confusion matrix
    cm = confusion_matrix(labels, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Generate the classification report
    report = classification_report(labels, y_pred)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    model_file = "svm_model.joblib"
    data_file = "preprocessed_data.npz"

    # Evaluate the model
    evaluate_model(model_file, data_file)