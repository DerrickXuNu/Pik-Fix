"""
data generation main function
"""
from utils.damage_libs import *
from utils.texture_libs import *
from utils.parser import data_parser
from hypes_yaml.yaml_utils import load_yaml

import os
import cv2
import time
import concurrent
from random import randint
from concurrent.futures import ThreadPoolExecutor


def process_single_image(image_full_name, input_folder, output_folder, hypes):
    """
    single image processing
    :param image_full_name:
    :param input_folder:
    :param output_folder:
    :param hypes:
    :return:
    """
    donwgrade = hypes['downgrade']
    upgrade = hypes['upgrade']
    crop_size = hypes['crop_size']
    dataset_name = hypes['dataset_name']

    image_name = image_full_name[:-4]
    extention = image_full_name[-3:]
    print(image_name)
    image = cv2.imread(input_folder + '/%s.%s' % (image_name, extention))

    # ignore gray images
    comparison = image[:, :, 0] == image[:, :, 1]
    equal_array = comparison.all()
    if equal_array:
        pass
    else:
        outputs = []
        h, w = image.shape[:2]

        # make sure the image size is larger than later on crop size
        new_crop_size = donwgrade // upgrade * crop_size
        if h < new_crop_size:
            image = cv2.resize(image, None, fx=new_crop_size / h, fy=new_crop_size / h)
        if w < new_crop_size:
            image = cv2.resize(image, None, fx=new_crop_size / w, fy=new_crop_size / w)
        h, w = image.shape[:2]

        # image should be able divided by donwgrade size without mod
        if h % donwgrade != 0:
            image = image[:h - h % donwgrade, :]
        if w % donwgrade != 0:
            image = image[:, :w - w % donwgrade]

        h, w = image.shape[:2]
        write_image = cv2.resize(image, (int(w // hypes['origin_downgrade']), int(h // hypes['origin_downgrade'])),
                                 interpolation=cv2.INTER_CUBIC)
        # ground truth
        cv2.imwrite(os.path.join(output_folder, '%s_%s.jpg' % (dataset_name, image_name)), write_image)

        # downgrade image
        image = cv2.resize(image, (int(w // donwgrade), int(h // donwgrade)),
                           interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (int(image.shape[1] * upgrade), int(image.shape[0] * upgrade)),
                           interpolation=cv2.INTER_CUBIC)

        outputs.append(image)

        # dust/crack effect
        count = 0
        for output in outputs:
            count += 1
            # crack generation
            code = randint(1, 12)
            processed_image, _ = crack_generate(output.copy(), code)
            # dust generation
            code = randint(1, 11)
            processed_image = dust_generate(processed_image.copy(), code)
            # damage generation
            processed_image = damage_generate(processed_image.copy(), threash=20)
            # convert to gray image
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(os.path.join(output_folder, '%s_%s_processed_%02d.jpg' % (dataset_name, image_name, count)),
                        processed_image)


def multiple_process(image_list, input_folder, output_folder, hypes):
    """
    Multi-threading
    :param image_list:
    :param input_folder:
    :param output_folder:
    :param hypes: yaml dictionary
    :return:
    """

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_image, image_full_name, input_folder, output_folder, hypes)
                   for image_full_name in image_list]
        concurrent.futures.wait(futures)


if __name__ == '__main__':
    opt = data_parser()
    yaml_file = opt.hypes_yaml
    hypes = load_yaml(yaml_file, opt)

    input_folder = hypes['input_folder']
    output_folder = hypes['output_folder']
    multi_thread = hypes['multi_thread']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_list_ = [x for x in os.listdir(input_folder) if 'process' not in x]
    existed_list = [x for x in os.listdir(output_folder) if 'process' not in x]

    if multi_thread:
        start_time = time.time()
        multiple_process(image_list_, input_folder, output_folder, hypes)
        duration1 = time.time() - start_time
        print('using multi-threading takes about %f' % duration1)
    else:
        start_time = time.time()
        for image_name in image_list_:
            if image_name not in existed_list:
                process_single_image(image_name, input_folder, output_folder, hypes)
        duration2 = time.time() - start_time
        print('single-threading takes about %f' % duration2)
