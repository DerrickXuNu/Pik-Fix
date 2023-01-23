"""Parser function"""
import argparse


def data_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True, help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument("--crack_net", action='store_true')
    parser.add_argument("--crack_dir", type=str, help='use this only when train colorization with pretrained '
                                                      'cracknet')
    parser.add_argument('--real_test', action='store_true')
    opt = parser.parse_args()
    return opt


def test_parser():
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument('--model_dir', type=str, required=True, help='model path')
    parser.add_argument('--input_path', type=str, required=True,
                        help='path to input image')
    parser.add_argument('--ref_path', type=str, required=True,
                        help='path to ref image')
    parser.add_argument('--crack_dir', type=str, help='crack net path')
    parser.add_argument('--real_test', action='store_true')

    opt = parser.parse_args()
    return opt

