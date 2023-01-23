import torch
import torch.nn as nn


class DCGAN(nn.Module):
    """
    DCGan Discriminator
    """

    def __init__(self, args):
        super(DCGAN, self).__init__()
        assert args['isize'] % 16 == 0, "isize has to be a multiple of 16"

        nc = args['nc']
        ndf = args['ndf']
        isize = args['isize']
        n_extra_layers = args['extra_layers']

        features = nn.Sequential()
        features.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        features.add_module('initial:{0}:relu'.format(ndf),
                            nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            features.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            if args['bn']:
                features.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                                    nn.BatchNorm2d(cndf))
            features.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                                nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            features.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            if args['bn']:
                features.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                                    nn.BatchNorm2d(out_feat))
            features.add_module('pyramid:{0}:relu'.format(out_feat),
                                nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        features.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                            nn.Conv2d(cndf, 1, int(csize), 1, 0, bias=False))
        self.features = features

    def forward(self, inputs):
        print (inputs.shape)
        output = self.features(inputs)
        print (output.shape)
        exit(0)
        output = output.mean(0)
        return output.view(1)
