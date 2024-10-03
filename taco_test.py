import torch
from torch.utils.data import DataLoader
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, postprocess
from efficientdet.dataset import CocoDataset, get_val_transform
import os
import json
from tqdm import tqdm
import yaml

def test_dataset(weights_path, test_dataset_path, compound_coef, num_workers=4, batch_size=1):
    # Settings
    threshold = 0.2
    iou_threshold = 0.2
    use_cuda = True
    use_float16 = False

    # Load your trained model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # Load test dataset
    test_dataset = CocoDataset(test_dataset_path, set='test', transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=test_dataset.collate_fn)

    results = []

    for iter, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            imgs = data['img']
            scales = data['scale']

            if use_cuda:
                imgs = imgs.cuda()

            imgs = imgs.float()
            if use_float16:
                imgs = imgs.half()

            features, regression, classification, anchors = model(imgs)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

            for i in range(len(out)):
                scale = scales[i]
                pred_boxes = out[i]['rois']
                pred_scores = out[i]['scores']
                pred_labels = out[i]['class_ids']

                for j in range(len(pred_boxes)):
                    pred_box = pred_boxes[j]
                    pred_score = pred_scores[j]
                    pred_label = pred_labels[j]

                    results.append({
                        'image_id': test_dataset.image_ids[iter * batch_size + i],
                        'category_id': test_dataset.label_to_coco_label(pred_label.item()),
                        'bbox': [
                            pred_box[0] / scale,
                            pred_box[1] / scale,
                            (pred_box[2] - pred_box[0]) / scale,
                            (pred_box[3] - pred_box[1]) / scale
                        ],
                        'score': float(pred_score)
                    })

    # Save results to a JSON file
    with open('test_results.json', 'w') as f:
        json.dump(results, f)

    print('Test completed. Results saved to test_results.json')

if __name__ == '__main__':
    weights_path = 'logs/taco/efficientdet-d0_74_9800.pth'  # Update this to your latest weights file
    test_dataset_path = 'datasets/taco/test'  # This should be the path to the directory containing your test images and annotations.json
    compound_coef = 0  # Change this to match your trained model
    
    # Load object list from your project configuration
    with open('projects/taco.yml', 'r') as f:
        config = yaml.safe_load(f)
        obj_list = config['obj_list']

    test_dataset(weights_path, test_dataset_path, compound_coef)