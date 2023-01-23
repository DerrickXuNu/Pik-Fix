""" a lib containing tools to generate crack/dust effect"""
import os
import cv2
import numpy as np
import imutils

from random import randint


def video2frame(video, output_folder):
    """
    Convert texture video to frames
    :param output_folder:
    :param video:
    :return:
    """
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_folder + "scratch_%03d.jpg" % count, image)
        success, image = vidcap.read()
        count += 1


def dust_generate(image, code):
    """
    dust effect generation
    :param image:
    :param code:
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    texutre = cv2.imread(os.path.join(dir_path, '../data/texture2/dust%02d.jpg' % code))

    if code in [8, 9, 10, 11]:
        texutre[texutre <= 50] = 0
    else:
        texutre[texutre <= 100] = 0

    mean_color = min(np.mean(image[image.shape[0] // 2, :])*3, 255)
    texutre[texutre != 0] = mean_color
    if image.shape[0] > texutre.shape[0] or image.shape[1] > texutre.shape[1]:
        texutre = cv2.resize(texutre, (image.shape[1], image.shape[0]))
    else:
        randx = randint(0, texutre.shape[0] - image.shape[0])
        randy = randint(0, texutre.shape[1] - image.shape[1])
        texutre = texutre[randx:randx+image.shape[0], randy:randy+image.shape[1]]

    image[texutre != 0] = texutre[texutre != 0]

    return image


def crack_generate(image, code):
    """
    Generate cracks
    :param code:
    :param image:
    :return: texture
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    texture = cv2.imread(os.path.join(dir_path,'../data/texture2/texture%02d.jpg' % code))
    if code == 12:
        texture[texture < 120] = 0

    # image rotation
    if code in [6, 8, 9, 10, 11, 12]:
        angle = randint(-30, 30)
        texture = imutils.rotate(texture, angle)

    # image trasnlation
    row, col = texture.shape[:2]
    random_list = [(randint(col // 4, col * 3 // 8), randint(row // 4, row * 3 // 8)),
                   (randint(col // 4, col * 3 // 8), randint(row // 8, row // 4)),
                   (randint(col // 8, col // 4), randint(row // 8, row // 4)),
                   (randint(col // 8, col // 4), randint(row // 4, row * 3 // 8))]
    seed = randint(0, 3)
    x_start, y_start = random_list[seed]
    texture = texture[y_start:y_start + row // 2, x_start:x_start + col // 2]

    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    image[texture > image] = texture[texture > image]

    return image, texture
