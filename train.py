"""
main function for training
"""
import os

from utils import parser, train_nogan
from hypes_yaml import yaml_utils

if __name__ == '__main__':
    # load training configuration from yaml file
    opt = parser.data_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # gpu setup
    use_gpu = hypes['train_params']['use_gpu']
    if use_gpu:
        # TODO: Multi-Gpu implementation
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypes['train_params']['gpu_id'])

    if opt.crack_net:
        # todo: add restoration training later
        pass

    elif 'gan' not in hypes:
        train_nogan.train(opt, hypes, use_gpu)

    else:
        # todo: add gan training later
        pass
        # train_gan.train(opt, hypes, use_gpu)