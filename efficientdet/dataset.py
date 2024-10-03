import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', self.set_name + '_annotations.coco.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        if set == 'train2017':
            self.transform = get_train_transform()
        else:
            self.transform = get_val_transform()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        annot = self.load_annotations(index)
        
        img_height, img_width = img.shape[:2]
        
        boxes = annot['bboxes']
        labels = annot['category_id']
        
        # Normalize bounding boxes
        normalized_boxes = [self.normalize_bbox(box, img_height, img_width) for box in boxes]
        
        transformed = self.transform(image=img, bboxes=normalized_boxes, category_ids=labels)
        
        img = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['category_ids']
        
        return {'img': img, 'annot': {'bboxes': boxes, 'category_id': labels}}

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def normalize_bbox(self, bbox, img_height, img_width):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        
        x1 = max(0, min(x1 / img_width, 1.0))
        y1 = max(0, min(y1 / img_height, 1.0))
        x2 = max(0, min(x2 / img_width, 1.0))
        y2 = max(0, min(y2 / img_height, 1.0))
        
        return [x1, y1, x2, y2]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomRotate90(p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.OneOf([
            A.RandomGamma(),
            A.HueSaturationValue(),
        ], p=0.3),
        A.ToGray(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

def get_val_transform():
    return A.Compose([
        A.Resize(height=640, width=640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
