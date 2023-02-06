"""
Helper functions for training and testing
"""
import os
import re
import yaml
import importlib
import glob
from datetime import datetime

import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms as transforms

from utils import loss
from datasets.OldPhotoDataset import *
from datasets.customized_transform import *


def setup_train(hypes):
    """
    create folder for saved model based on current timestep and model name
    :param hypes: config yaml dictionary for training
    :return:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')
    full_path = os.path.join(current_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def setup_train_crack(hypes, model_name='cracknet_rdn'):
    """
    create folder for saved model based on current timestep and model name
    :param hypes: config yaml dictionary for training
    :return:
    """
    model_name = model_name
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')
    full_path = os.path.join(current_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_dataset(hypes, train=True, gan=False, real=False, crack_dir=None):
    """
    create customized Datasets
    :param gan: Whether for gan training
    :param hypes: config yaml file
    :param train: flag whether to train or test
    :return:
    """
    if real:
        dataset = RealOldPhotoDataset(hypes['real_file'],
                                      transform=transforms.Compose(
                                          [TolABTensor()]))
        # in case the users collect more old photo pairs and want to use those for training
        if train:
            dataset = RealOldPhotoDataset(hypes['real_file'],
                                          transform=transforms.Compose(
                                              [RandomCrop(256),
                                               TolABTensor()]))
            loader_train = DataLoader(dataset,
                                      batch_size=hypes['gan'][
                                          'batch_size'] if gan else
                                      hypes['train_params'][
                                          'batch_size'],
                                      shuffle=True,
                                      num_workers=4)
            return loader_train, loader_train
        else:
            return dataset

    if train:
        # if we only train the color restoration part
        if not crack_dir:
            transform_operation = transforms.Compose([RandomCrop(256),
                                                      TolABTensor()])
        else:
            transform_operation = transforms.Compose([
                RandomBlur(),
                CrackGenerator(),
                RandomCrop(256),
                TolABTensor()])

        train_dataset = OldPhotoDataset(hypes['train_file'],
                                        transform=transform_operation,
                                        ref_json=hypes['train_params'][
                                            'ref_json'])
        loader_train = DataLoader(train_dataset,
                                  batch_size=hypes['gan'][
                                      'batch_size'] if gan else
                                  hypes['train_params']['batch_size'],
                                  shuffle=True,
                                  num_workers=4)

        val_dataset = OldPhotoDataset(hypes['val_file'],
                                      transform=transforms.Compose(transform_operation))
        loader_val = DataLoader(val_dataset, batch_size=1, shuffle=False)

        return loader_train, loader_val

    else:
        if not crack_dir:
            transform_operation = transforms.Compose(TolABTensor())
        else:
            transform_operation = transforms.Compose([
                RandomBlur(),
                CrackGenerator(),
                TolABTensor()])
        test_dataset = OldPhotoDataset(root_dir=hypes['test_file'],
                                       transform=transform_operation)
        return test_dataset


def create_model(hypes, dis=False, crack=False):
    """
    Import the module "models/[model_name].py
    :param crack: indicate whether this is for cracknet
    :param dis: whether this is a discriminator
    :param hypes:
    :return:
    """
    if dis:
        backbone_name = hypes['gan']['discrimiator']['arch']['backbone']
        backbone_config = hypes['gan']['discrimiator']['arch']['args']
    elif crack:
        backbone_name = hypes['crack_arch']['backbone']
        backbone_config = hypes['crack_arch']['args']
    else:
        backbone_name = hypes['arch']['backbone']
        backbone_config = hypes['arch']['args']

    model_filename = "models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            'backbone not found in models folder. Please make sure you have a python file named %s and has a class'
            'called %s ignoring upper/lower case' % (
            model_filename, target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def print_network(net):
    """
    Print the layers and number of params of the model
    :param net: neural network object
    :return:
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def setup_loss(hypes):
    """
    Setup loss functions
    :param hypes:
    :return:
    """
    criterion = {}
    for name, value in hypes['train_params']['loss'].items():
        loss_func = getattr(loss, name)(value['args'])
        if hypes['train_params']['use_gpu']:
            loss_func.cuda()
        criterion[name] = loss_func

    if 'intermediate' in hypes['train_params']:
        for name, value in hypes['train_params']['intermediate'][
            'loss'].items():
            loss_func = getattr(loss, name)(value['args'])
            if hypes['train_params']['use_gpu']:
                loss_func.cuda()
            criterion['intermediate_' + name] = loss_func

    return criterion


def setup_gan_loss(hypes, dis=False):
    """
    Setup loss function for gan training
    :param hypes:
    :param dis:
    :return:
    """
    criterion = {}
    if not dis:
        criterion = setup_loss(hypes)
    for name, value in hypes['gan']['loss'].items():
        loss_func = getattr(loss, name)(value['args'])
        if hypes['train_params']['use_gpu']:
            loss_func.cuda()
        criterion.update({name: loss_func})

    return criterion


def setup_optimizer(method, model):
    """
    Create optimizer corresponding to the yaml file
    :param model:
    :param name:
    :return:
    """
    method_dict = method
    optimizer_method = getattr(optim, method_dict['name'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'params' in method_dict:
        return optimizer_method(model.parameters(), lr=method_dict['lr'],
                                **method_dict['params'])
    else:
        return optimizer_method(model.parameters(), lr=method_dict['lr'])


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted
    :param saved_path:  model saved path, str
    :param model:  model object
    :return:
    """
    if not os.path.exists(saved_path):
        raise ValueError('{} not found'.format(saved_path))

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(saved_path,
                                                      'net_epoch%d.pth' %
                                                      initial_epoch)))

    return initial_epoch, model


def log_images(input_l, input_batch, ref_ab, ref_gray, writer, model, epoch,
               att_model):
    """
    write the images to tensorboard for visualization
    :param input_data:  input image
    :param target_data:  groundtruth image
    :param writer:  SummaryWriter
    :param model:  trained model
    :param epoch:  current epoch
    :return:
    """
    model.eval()
    output_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
    output = torch.clamp(output_dict['output'], -1., 1.)

    output = lab_to_rgb(input_l, output).cuda()

    im_input = utils.make_grid(input_batch.data, nrow=8, normalize=True,
                               scale_each=True)
    im_target = utils.make_grid(ref_gray.data, nrow=8, normalize=True,
                                scale_each=True)
    im_restore = utils.make_grid(output.data, nrow=8, normalize=True,
                                 scale_each=True)

    writer.add_image('input image', im_input, epoch + 1)
    writer.add_image('groundtruth image', im_target, epoch + 1)
    writer.add_image('restored image', im_restore, epoch + 1)

    return writer


def log_images_crack(input_l, gt_l, writer, model, epoch):
    """
    write the images to tensorboard for visualization
    :param input_data:  input image
    :param target_data:  groundtruth image
    :param writer:  SummaryWriter
    :param model:  trained model
    :param epoch:  current epoch
    :return:
    """
    model.eval()
    output_dict = model(input_l)
    output = torch.clamp(output_dict['output'], -1., 1.)

    output = (output + 1.) / 2.
    gt_l = (gt_l + 1.) / 2.
    input_l = (input_l + 1.) / 2.

    im_target = utils.make_grid(gt_l.data, nrow=8, normalize=True,
                                scale_each=True)
    im_restore = utils.make_grid(output.data, nrow=8, normalize=True,
                                 scale_each=True)
    im_input = utils.make_grid(input_l.data, nrow=8, normalize=True,
                               scale_each=True)

    writer.add_image('groundtruth image', im_target, epoch + 1)
    writer.add_image('input image', im_input, epoch + 1)
    writer.add_image('restored image', im_restore, epoch + 1)

    return writer


def val_eval(model, att_model, loader_val, writer, opt, epoch, crack_net):
    """
    evaluate on validation dataset
    :param epoch:  current training epoch
    :param model:  trained model
    :param loader_val:  pytorch data loader for validaion dataset
    :param writer:  summary writer
    :param opt:  training option
    :param crack_net: crack net
    :return:
    """
    model.eval()
    count = 0
    psnr = 0
    for j, batch_data in enumerate(loader_val):
        input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab = batch_data[
                                                                  'input_image'], \
                                                              batch_data[
                                                                  'input_L'], \
                                                              batch_data[
                                                                  'gt_ab'], \
                                                              batch_data[
                                                                  'gt_L'], \
                                                              batch_data[
                                                                  'ref_gray'], \
                                                              batch_data[
                                                                  'ref_ab']

        input_batch = input_batch.cuda()
        input_l = input_l.cuda()
        gt_ab = gt_ab.cuda()
        gt_l = gt_l.cuda()
        ref_gray = ref_gray.cuda()
        ref_ab = ref_ab.cuda()

        if opt.crack_dir:
            input_l = crack_net(input_l)['output']

        out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
        output = torch.clamp(out_dict['output'], -1, 1.)
        output = lab_to_rgb(input_l, output).cuda()

        target_val = lab_to_rgb(gt_l, gt_ab).cuda()

        psnr += loss.batch_psnr(output, target_val, 1.)
        count += 1

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('At current epoch %d, the psnr on validation dataset is %f' % (
    epoch, psnr / count))
    writer.add_scalar('PSNR on Val', psnr, epoch)

    return writer


def val_eval_crack(model, loader_val, writer, epoch):
    """
    evaluate on validation dataset
    :param epoch:  current training epoch
    :param model:  trained model
    :param loader_val:  pytorch data loader for validaion dataset
    :param writer:  summary writer
    :param hypes:  train config yaml file
    :return:
    """
    model.eval()
    count = 0
    psnr = 0
    for j, batch_data in enumerate(loader_val):
        input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab = batch_data[
                                                                  'input_image'], \
                                                              batch_data[
                                                                  'input_L'], \
                                                              batch_data[
                                                                  'gt_ab'], \
                                                              batch_data[
                                                                  'gt_L'], \
                                                              batch_data[
                                                                  'ref_gray'], \
                                                              batch_data[
                                                                  'ref_ab']

        input_l = input_l.cuda()
        gt_l = gt_l.cuda()

        out_dict = model(input_l)
        output = torch.clamp(out_dict['output'], -1, 1.)

        output = (output + 1.) / 2.
        gt_l = (gt_l + 1.) / 2.

        psnr += loss.batch_psnr(output, gt_l, 1.)
        count += 1

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('At current epoch %d, the psnr on validation dataset is %f' % (
    epoch, psnr / count))
    writer.add_scalar('PSNR on Val', psnr, epoch)

    return writer


def write_test(output, model_path, image_name):
    """
    Write output image to model saved path
    :param output: pytorch tensor
    :param model_path: saved model path
    :param image_name: indicate image order
    :return:
    """
    output_folder = os.path.join(model_path, 'test_images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_out = np.uint8(255 * output.data.cpu().numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(output_folder, '%s.jpg' % image_name), save_out)
