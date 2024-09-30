# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/{val_set_name}_annotations.coco.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    batch_size = 1  # Process one image at a time
    for i in tqdm(range(0, len(image_ids), batch_size)):
        batch_ids = image_ids[i:i+batch_size]
        batch_imgs = []
        batch_metas = []

        for image_id in batch_ids:
            image_info = coco.loadImgs(image_id)[0]
            image_path = img_path + image_info['file_name']

            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
            batch_imgs.extend(framed_imgs)
            batch_metas.extend(framed_metas)

        x = torch.stack([torch.from_numpy(fi) for fi in batch_imgs], 0)

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, nms_threshold)

        print(f"Processing batch {i+1}/{len(image_ids)//batch_size}")
        print(f"out length: {len(out)}")
        for j, image_id in enumerate(batch_ids):
            print(f"Processing image {j+1}/{len(batch_ids)}, image_id: {image_id}")
            if j >= len(out):
                print(f"Warning: j ({j}) is out of range for out (length {len(out)})")
                continue
            
            print(f"out[{j}] keys: {out[j].keys()}")
            if 'rois' not in out[j]:
                print(f"Warning: 'rois' not in out[{j}]")
                continue
            
            if len(out[j]['rois']) == 0:
                print(f"No predictions for image {image_id}")
                continue

            preds = invert_affine(batch_metas[j], out[j])
            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            if rois.shape[0] > 0:
                # x1,y1,x2,y2 -> x1,y1,w,h
                rois[:, 2] -= rois[:, 0]
                rois[:, 3] -= rois[:, 1]

                bbox_score = scores

                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    label = int(class_ids[roi_id])
                    box = rois[roi_id, :]

                    image_result = {
                        'image_id': image_id,
                        'category_id': label + 1,
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)

        # Clear GPU cache
        torch.cuda.empty_cache()

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/{SET_NAME}_annotations.coco.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 300  # Limit the number of images to evaluate
    coco_gt = COCO(VAL_GT)

    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    # Get the latest weight file
    weight_files = [f for f in os.listdir(f'logs/{project_name}') if f.endswith('.pth')]
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in logs/{project_name}")
    latest_weight = max(weight_files, key=lambda f: os.path.getmtime(os.path.join(f'logs/{project_name}', f)))
    weights_path = os.path.join(f'logs/{project_name}', latest_weight)

    print(f'Using weights: {weights_path}')

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    if use_cuda:
        model.cuda(gpu)

    model.eval()

    evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')