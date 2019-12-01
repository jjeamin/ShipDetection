import torch
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None):
        for trans in self.transforms:
            image, boxes = trans(image, boxes)

        return image, boxes


class ToTensor(object):
    def __call__(self, image, targets=None):
        """
        :arg
        image : (numpy)
        targets : (list) [[(numpy) box, class], [[(numpy) box, class]]

        :return
        numpy : H x W x C -> tensor : C x H x W Float32
        """

        image = image / 255.0
        image = torch.from_numpy(image).permute((2, 0, 1))

        return image, targets


class Resize(object):
    def __init__(self, output_size):
        """
        output_size: (H, W)
        """
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image, targets=None):
        """
        image : (numpy)
        targets : (list) [[(numpy) box, class], [[(numpy) box, class]]
        """
        h, w = image.shape[:2]

        new_w, new_h = self.output_size

        new_w, new_h = int(new_w), int(new_h)
        scale_w, scale_h = float(new_w) / w, float(new_h) / h

        image_trans = cv2.resize(image, (new_w, new_h,))
        for i, t in enumerate(targets):
            targets[i][0] = targets[i][0] * [scale_w, scale_h, scale_w, scale_h, scale_w, scale_h, scale_w, scale_h]

        return image_trans, targets