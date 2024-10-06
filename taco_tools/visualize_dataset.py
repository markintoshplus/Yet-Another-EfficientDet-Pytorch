import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load the annotation file
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Draw bounding boxes and labels on the image
def draw_boxes_and_labels(image, annotations, categories):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        category_id = ann['category_id']
        category_name = next(cat['name'] for cat in categories if cat['id'] == category_id)
        
        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
        draw.text((x, y-10), category_name, fill='red', font=font)
    
    return image

# Visualize random images with their annotations
def visualize_dataset(annotations, image_dir, num_samples=5):
    # Check if annotations is a list or a dictionary
    if isinstance(annotations, list):
        # If it's a list, we need to process it differently
        categories = {cat['id']: cat['name'] for cat in annotations[0]['categories']}
        images = annotations[0]['images']
        annotations = annotations[0]['annotations']
    else:
        # If it's a dictionary, proceed as before
        categories = annotations['categories']
        images = annotations['images']
        annotations = annotations['annotations']
    
    # Group annotations by image_id
    ann_dict = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in ann_dict:
            ann_dict[image_id] = []
        ann_dict[image_id].append(ann)
    
    # Randomly select images
    sample_images = random.sample(images, num_samples)
    
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 4))
    for i, img_info in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = Image.open(img_path)
        
        # Draw bounding boxes and labels
        img_anns = ann_dict.get(img_info['id'], [])
        img_with_boxes = draw_boxes_and_labels(img.copy(), img_anns, categories)
        
        axs[i].imshow(img_with_boxes)
        axs[i].axis('off')
        axs[i].set_title(f"Image ID: {img_info['id']}")
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Update these paths according to your dataset location
    annotation_dir = 'datasets/taco/annotations'
    image_dir = 'datasets/taco'
    
    # Visualize for train, val, and test sets
    for split in ['train', 'val', 'test']:
        print(f"Visualizing {split} set:")
        annotation_file = os.path.join(annotation_dir, f'{split}_annotations.coco.json')
        split_image_dir = os.path.join(image_dir, split)
        
        annotations = load_annotations(annotation_file)
        visualize_dataset(annotations, split_image_dir)