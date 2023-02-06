import os

import torch
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
from torchvision.models import resnet34, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck

from utils import helper, loss
from utils.color_space_convert import lab_to_rgb
from models.networks import AttentionExtractModule


def train(opt, hypes, use_gpu=True):
    """
    Train model if there is no gan
    :param opt: arg parse
    :param use_gpu: whether use gpu
    :param hypes:  dictionary of training params
    :return:
    """

    print('loading dataset')
    loader_train, loader_val = helper.create_dataset(hypes,
                                                     train=True,
                                                     gan=False,
                                                     real=opt.real_test,
                                                     crack_dir=opt.crack_dir)

    print('creating model')
    # pretrained resnet for attention extraction
    base_resnet = resnet34(pretrained=True)
    att_model = AttentionExtractModule(BasicBlock, [3, 4, 6, 3])
    att_model.load_state_dict(base_resnet.state_dict())
    att_model.eval()

    # crack net to refine L channel
    crack_net = helper.create_model(hypes, crack=True)
    model = helper.create_model(hypes)

    helper.print_network(model)
    if use_gpu:
        att_model.cuda()
        crack_net.cuda()
        model.cuda()

    # define the loss criterion
    criterion = helper.setup_loss(hypes)

    # optimizer setup
    optimizer = helper.setup_optimizer(hypes['train_params']['solver'], model)
    # default schedular use exponential schedule
    # #TODO: Make this parameterized in yaml file
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # load pretrained cracknet
    if opt.crack_dir:
        _, crack_net = helper.load_saved_model(opt.crack_dir, crack_net)

    # load saved model for continue training or train from scratch
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = helper.load_saved_model(saved_path, model)
    else:
        # setup saved model folder
        init_epoch = 0
        saved_path = helper.setup_train(hypes)
    # record training
    writer = SummaryWriter(saved_path)
    crack_net.eval()

    print('training start')
    epoches = hypes['train_params']['epoches']
    step = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        for i, batch_data in enumerate(loader_train):
            # clean up grad first
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab = batch_data['input_image'], \
                                                                   batch_data['input_L'], \
                                                                   batch_data['gt_ab'], batch_data['gt_L'], \
                                                                   batch_data['ref_gray'], batch_data['ref_ab']
            if use_gpu:
                input_batch = input_batch.cuda()
                input_l = input_l.cuda()
                gt_ab = gt_ab.cuda()
                gt_l = gt_l.cuda()
                ref_gray = ref_gray.cuda()
                ref_ab = ref_ab.cuda()
            # if the cracknet is also involved, then use it to restore
            # the image first
            if opt.crack_dir:
                input_l = crack_net(input_l)['output']

            # model inference and loss cal
            out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
            final_loss = loss.loss_sum(hypes, criterion, out_dict, gt_ab)

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # plot and print training info
            if step % hypes['train_params']['display_freq'] == 0:
                model.eval()
                out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
                out_train = torch.clamp(out_dict['output'], -1., 1.)

                out_train = lab_to_rgb(input_l, out_train).cuda()
                target_train = lab_to_rgb(gt_l, gt_ab).cuda()

                psnr_train = loss.batch_psnr(out_train, target_train, 1.)
                print("[epoch %d][%d/%d], total loss: %.4f, PSNR: %.4f" % (epoch + 1, i + 1, len(loader_train),
                                                                           final_loss.item(), psnr_train))
                writer.add_scalar('generator pretrain loss', final_loss.item(), step)
                writer.add_scalar('PSNR during pretrain', psnr_train, step)
            step += 1

        # log images
        writer = helper.log_images(input_l, input_batch, ref_ab, ref_gray, writer, model, epoch, att_model)

        # evaluate model on validation dataset
        if epoch % hypes['train_params']['eval_freq'] == 0:
            writer = helper.val_eval(model, att_model, loader_val, writer, opt, epoch, crack_net)

        if epoch % hypes['train_params']['writer_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

    return model, att_model, crack_net, writer
