import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import CocoDataset, Resizer, Normalizer

def visualize_dataset(root_dir, set_name='train', num_samples=5):
    # Initialize the dataset
    dataset = CocoDataset(root_dir=root_dir, set=set_name, transform=None)
    
    # Create a figure to display images
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    
    for i in range(num_samples):
        # Get a random sample from the dataset
        sample = dataset[np.random.randint(0, len(dataset))]
        img, annot = sample['img'], sample['annot']
        
        # Convert the image back to uint8 and BGR format for visualization
        img_vis = (img * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes on the image
        for bbox in annot:
            x1, y1, x2, y2, label = bbox
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_vis, dataset.labels[int(label)], (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image
        axes[i].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i+1}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the root directory of your dataset
    root_dir = "datasets/taco"
    
    # Visualize the dataset
    visualize_dataset(root_dir, set_name='train', num_samples=5)