import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

def load_data(data_file):
    data = np.load(data_file)
    return data['image_data'], data['labels']

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_and_visualize(model_file, data_file, class_names):
    # Load the preprocessed data
    image_data, labels = load_data(data_file)

    # Load the trained model
    model = joblib.load(model_file)

    # Make predictions on the entire dataset
    y_pred = model.predict(image_data)

    # Ensure labels and predictions are integers
    labels = labels.astype(int)
    y_pred = y_pred.astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(labels, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix as a heatmap
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    model_file = "svm_model.joblib"
    data_file = "preprocessed_data.npz"
    class_names = ["Aculus olearius", "Olive peacock spot", "Healthy"]  # Replace with your actual class names

    # Evaluate the model and visualize the confusion matrix
    evaluate_and_visualize(model_file, data_file, class_names)