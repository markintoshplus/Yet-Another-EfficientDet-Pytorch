import json
from collections import defaultdict
import os

def analyze_bboxes(annotation_files):
    aspect_ratios = defaultdict(int)
    scales = defaultdict(int)
    
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        for ann in data['annotations']:
            bbox = ann['bbox']
            w, h = bbox[2], bbox[3]
            aspect_ratio = w / h if h != 0 else 0
            scale = (w * h) ** 0.5
            
            aspect_ratios[round(aspect_ratio, 1)] += 1
            scales[round(scale, -1)] += 1
    
    return aspect_ratios, scales

# Paths to your annotation files
annotation_dir = 'datasets/taco/annotations'
annotation_files = [
    os.path.join(annotation_dir, 'train_annotations.coco.json'),
    os.path.join(annotation_dir, 'test_annotations.coco.json'),
    os.path.join(annotation_dir, 'val_annotations.coco.json')
]

aspect_ratios, scales = analyze_bboxes(annotation_files)
print("Aspect Ratios:", dict(aspect_ratios))
print("Scales:", dict(scales))

# Suggest anchor ratios and scales
suggested_ratios = sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)[:3]
suggested_scales = sorted(scales.items(), key=lambda x: x[1], reverse=True)[:3]

print("\nSuggested anchor ratios:", [ratio for ratio, _ in suggested_ratios])
print("Suggested anchor scales:", [scale for scale, _ in suggested_scales])
