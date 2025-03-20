import numpy as np
import joblib

def save_model(model, model_file):
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

def load_model(model_file):
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
    return model

def make_inference(model, input_data):
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Example usage
    from sklearn.svm import SVC

    # Create a sample model (for demonstration purposes)
    sample_model = SVC(kernel='linear')
    sample_data = np.random.rand(10, 64*64*3)  # Random data for demonstration
    sample_labels = np.random.randint(0, 2, 10)  # Random labels for demonstration
    sample_model.fit(sample_data, sample_labels)

    # Save the model
    model_file = "sample_model.joblib"
    save_model(sample_model, model_file)

    # Load the model
    loaded_model = load_model(model_file)

    # Make inference
    test_data = np.random.rand(5, 64*64*3)  # Random test data for demonstration
    predictions = make_inference(loaded_model, test_data)
    print(f"Predictions: {predictions}")