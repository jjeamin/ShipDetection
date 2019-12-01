import cv2
import os
import json
import numpy as np
import torch
import torch.utils.data as data

TRAIN_IMAGE_PATH = './datasets/train/images/'
TEST_IMAGE_PATH = './datasets/test/images/'
TRAIN_LABEL_PATH = './datasets/train/labels.json'
TEST_LABEL_PATH = './datasets/test/labels.json'

def custom_collate(batch):
    targets = [x[1] for x in batch]
    images = [x[0] for x in batch]

    return torch.stack(images, 0), targets


class CustomDataset(data.Dataset):
    def __init__(self,
                 dataType='train',
                 transform=None,
                 torch_transform=None):
        self.file_list = os.listdir(TRAIN_IMAGE_PATH)
        self.imgs = [os.path.join(TRAIN_IMAGE_PATH, x) for x in self.file_list]
        self.labels = self.parse_json(TRAIN_LABEL_PATH)
        self.transform = transform
        self.torch_transform = torch_transform

    def __getitem__(self, index):
        '''
        :param
        index : random index
        :return:
        image : (numpy)
        targets : (list) [(numpy), (numpy)]
        '''
        img = cv2.imread(self.imgs[index])
        targets = self.labels[int(self.file_list[index][:-4])]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img, targets = self.transform(img, targets)

        if self.torch_transform is not None:
            img = self.torch_transform(img)

        return img, targets

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def parse_json(label_path):
        res = {}

        datas = json.load(open(label_path))

        for d in datas["features"]:
            properties = d["properties"]

            img_id = int(properties["image_id"][:-4])
            box = properties["bounds_imcoords"].split(',')
            cls = properties["type_id"]

            box = np.array([float(b) for b in box])
            if img_id not in res:
                res[img_id] = []

            res[img_id] += [[box, cls]]

        return res


