"""
A lib containing tools for generating damage effect
"""
import cv2
import numpy as np
from scipy import ndimage
from random import randint


def seed_point(h, w, seed):
    """
    Generate edge point based on seed
    :param seed:
    :param h:
    :param w:
    :return:
    """
    if seed == 1:
        x = 0
        y = randint(1, h - 5)
    elif seed == 2:
        y = h - 1
        x = randint(1, w - 5)
    elif seed == 3:
        x = w - 1
        y = randint(1, h - 5)
    elif seed == 4:
        y = 0
        x = randint(1, w - 5)

    point = (x, y)

    return point


def cal_area(point_1, point_2, seed1, seed2, h, w, threash=5):
    """
    calculate the rough area of the damage
    :param threash:
    :param w:
    :param h:
    :param seed2:
    :param seed1:
    :param point_1:
    :param point_2:
    :return:
    """
    area = 0
    if seed1 == 1:
        if seed2 == 2:
            edge1 = h - point_1[1]
            edge2 = point_2[0]
            area = edge1 * edge2 / 2
        if seed2 == 3:
            if (point_1[1] <= h // threash and point_2[1] <= h // threash) or (
                    point_1[1] >= h - h//threash and point_2[1] >= h - h//threash):
                area = 0
            else:
                area = h * w
        if seed2 == 4:
            edge1 = point_1[1]
            edge2 = point_2[0]
            area = edge1 * edge2 / 2

    if seed1 == 2:
        if seed2 == 1:
            edge1 = point_1[0]
            edge2 = h - point_2[1]
            area = edge1 * edge2 / 2
        if seed2 == 3:
            edge1 = w - point_1[0]
            edge2 = h - point_2[1]
            area = edge1 * edge2 / 2
        if seed2 == 4:
            if (point_1[0] <= w // threash and point_2[0] <= w // threash) or (
                    point_1[0] >= w - w // threash and point_2[0] >= w - w // threash):
                area = 0
            else:
                area = h * w

    if seed1 == 3:
        if seed2 == 1:
            if (point_1[1] <= h // threash and point_2[1] <= h // threash) or (
                    point_1[1] >= h - h // threash and point_2[1] >= h - h // threash):
                area = 0
            else:
                area = h * w
        if seed2 == 2:
            edge1 = h - point_1[1]
            edge2 = w - point_2[0]
            area = edge2 * edge1 / 2
        if seed2 == 4:
            edge1 = point_1[1]
            edge2 = w - point_2[0]
            area = edge1 * edge2 / 2

    if seed1 == 4:
        if seed2 == 1:
            edge1 = point_1[0]
            edge2 = point_2[1]
            area = edge1 * edge2 / 2
        if seed2 == 2:
            if (point_1[0] <= w // threash and point_2[0] <= w // threash) or (
                    point_1[0] >= w - w // threash and point_2[0] >= w - w // threash):
                area = 0
            else:
                area = h * w
        if seed2 == 3:
            edge1 = w - point_1[0]
            edge2 = point_2[1]
            area = edge1 * edge2 / 2
    return area


def mask_generate(point_1, point_2, seed1, seed2, mask, threash=5):
    """
    generate polygon mask area
    :param threash:
    :param point_1:
    :param point_2:
    :param seed1:
    :param seed2:
    :param mask:
    :return:
    """
    h, w = mask.shape[:2]
    points = [point_1, point_2]

    if seed1 in [1, 2] and seed2 in [1, 2]:
        points.append((0, h - 1))

    if seed1 in [1, 3] and seed2 in [1, 3]:
        if point_1[1] <= h / threash:
            if point_1[0] < point_2[0]:
                points.append((w - 1, 0))
                points.append((0, 0))
            else:
                points.append((0, 0))
                points.append((w - 1, 0))
        else:
            if point_1[0] < point_2[0]:
                points.append((w - 1, h - 1))
                points.append((0, h - 1))
            else:
                points.append((0, h - 1))
                points.append((w - 1, h - 1))

    if seed1 in [1, 4] and seed2 in [1, 4]:
        points.append((0, 0))

    if seed1 in [2, 3] and seed2 in [2, 3]:
        points.append((w - 1, h - 1))

    if seed1 in [2, 4] and seed2 in [2, 4]:
        if point_1[0] <= w / threash:
            if point_1[1] < point_2[1]:
                points.append((0, h - 1))
                points.append((0, 0))
            else:
                points.append((0, 0))
                points.append((0, h - 1))
        else:
            if point_1[1] < point_2[1]:
                points.append((w - 1, h - 1))
                points.append((w - 1, 0))
            else:
                points.append((w - 1, 0))
                points.append((w - 1, h - 1))
    if seed1 in [3, 4] and seed2 in [3, 4]:
        points.append((w - 1, 0))

    points = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    return mask, points


def hull_points(points, central_point):
    """
    Generate final points for hull mask
    :param points: edge points
    :param central_point:  central point of mask
    :return:
    """
    interp_points = np.linspace(points[0], points[1], abs(points[0, 0] - points[1, 0]))
    valid_num = randint(interp_points.shape[0] // 5, max(interp_points.shape[1] // 4, interp_points.shape[0] // 5 + 3))
    valid_num = min(interp_points.shape[0], valid_num)
    idx = np.random.randint(interp_points.shape[0], size=valid_num)
    idx = np.sort(idx)
    valid_points = interp_points[idx]

    final_points = [tuple(points[0])]
    for j in range(valid_points.shape[0]):
        point = valid_points[j]
        new_point = (central_point + point) // 2
        final_points.append(tuple(new_point))

    final_points.append(tuple(points[1]))
    final_points = np.vstack((final_points, points[2:]))
    final_points = np.asarray(final_points, dtype=np.int32)

    return final_points


def damage_generate(image, threash=5):
    """
    simulate the old photo damage effect
    :param image:
    :return:
    """
    h, w = image.shape[:2]
    mask1 = 255 * np.ones((h, w), dtype=np.uint8)

    # middle part should not have damage
    x1 = w // 5
    x2 = 4 * w // 5
    y1 = h // 5
    y2 = 4 * h // 5
    mask1[y1:y2, x1:x2] = 0

    # damage should be too large
    area = h * w
    while area >= h * w // threash:
        mask2 = np.zeros_like(mask1)
        # create start and end point on the edge
        seed1 = randint(1, 4)
        seed2 = randint(1, 4)
        # it is very rare to see the damage only on one edge
        while seed1 == seed2:
            seed2 = randint(1, 4)
        point_1 = seed_point(h, w, seed1)
        point_2 = seed_point(h, w, seed2)
        area = cal_area(point_1, point_2, seed1, seed2, h, w, threash=threash)

    # generate the mask
    mask2, points = mask_generate(point_1, point_2, seed1, seed2, mask2, threash=threash)
    mask = cv2.bitwise_and(mask1, mask2)

    # get random points within the mask for drawing polygon
    polygon_all_points = np.argwhere(mask == 255)
    polygon_all_points = np.flip(polygon_all_points, axis=1)
    # central point
    central_point = np.mean(polygon_all_points, axis=0)
    final_points = hull_points(points, central_point)
    # fill poly
    output = image.copy()
    cv2.fillPoly(output, [final_points], (255, 255, 255))

    return output


if __name__ == '__main__':
    image_name = '../data/outputs/2007_000027_processed_01.jpg'
    image = cv2.imread(image_name)
    for i in range(200):
        output = damage_generate(image, 20)
        cv2.imshow('final', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
