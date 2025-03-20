import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_images(image_dir, output_file):
    image_data = []
    labels = []
    class_names = ["aculus_olearius", "olive_peacock_spot", "healthy"]
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                img = cv2.imread(os.path.join(class_dir, filename))
                if img is not None:
                    # Resize the image to 64x64
                    resized_img = cv2.resize(img, (64, 64))
                    # Flatten the image into a feature vector
                    flattened_img = resized_img.flatten()
                    image_data.append(flattened_img)
                    # Append the class label
                    labels.append(class_map[class_name])
                else:
                    print(f"Warning: Unable to read image {filename}")

    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)

    # Check if image_data is not empty
    if image_data.size == 0:
        raise ValueError("No images were processed. Please check the image directory and ensure it contains valid images.")

    # Standardize the feature vectors
    scaler = StandardScaler()
    image_data = scaler.fit_transform(image_data)

    # Save the preprocessed data to a file
    np.savez(output_file, image_data=image_data, labels=labels)
    print(f"Preprocessed data saved to {output_file}")
    print(f"Number of samples: {len(image_data)}")
    print(f"Class distribution: {np.bincount(labels)}")

if __name__ == "__main__":
    image_dir = "C:\\Users\\talal\\KypriaElia\\images"  # Replace with the actual path to your image directory
    output_file = "preprocessed_data.npz"
    preprocess_images(image_dir, output_file)