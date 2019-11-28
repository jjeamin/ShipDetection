import cv2
import numpy as np


def save_boxing_image(image, targets):
    '''
    :param image: (tensor) cpu image
    :return: (file) save image
    '''
    image = image.permute(1, 2, 0).numpy() * 255.0
    image = image.astype('uint8')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    for target in targets:
        target = target[0].astype(np.int)

        image = cv2.line(image, (target[0], target[1]), (target[2], target[3]),
                         (255, 0, 0), 3)
        image = cv2.line(image, (target[2], target[3]), (target[4], target[5]),
                         (255, 0, 0), 3)
        image = cv2.line(image, (target[4], target[5]), (target[6], target[7]),
                         (255, 0, 0), 3)
        image = cv2.line(image, (target[6], target[7]), (target[0], target[1]),
                         (255, 0, 0), 3)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./test.png', image)