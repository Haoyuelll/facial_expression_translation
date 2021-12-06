import numpy as np
import torch
from torch import tensor
from torch._C import device
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import cv2
import face_alignment as F
import os
from PIL import Image


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        os.environ['TORCH_HOME'] = "/home6/liuhy/torch_home"
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        # if opt.nce_idt and self.isTrain:
        #     self.loss_names += ['NCE_Y']
        #     self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        from . import perceptual_model
        self.vgg_perceptual = perceptual_model.VGG16_for_Perceptual()
        self.vgg_perceptual.to('cuda:2')

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        # self.bbox_A = self.bbox_A[:bs_per_gpu]
        # self.bbox_B = self.bbox_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.bbox_A = input['A_mask' if AtoB else 'B_mask'].to(self.device)
        self.bbox_B = input['B_mask' if AtoB else 'A_mask'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]   # netG(real_A)
        # if self.opt.nce_idt:
        #     self.idt_B = self.fake[self.real_A.size(0):]  # netG(real_B)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            # self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            # loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) / 2

            self.masked_L1_loss = self.calculate_masked_loss(self.real_A, self.fake_B, self.bbox_A)
            loss_NCE_both = (self.loss_NCE * 10 + self.masked_L1_loss) / 11
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        # feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        # feat_k = self.netG(src, self.nce_layers, encode_only=True)
        # feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        # feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        feat_k = [src]
        for h in self.vgg_perceptual.forward(src):
            feat_k.append(h)
            # print(h)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        # print(feat_k_pool)

        feat_q = [tgt]
        for h in self.vgg_perceptual.forward(tgt):
            feat_q.append(h)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0

        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def return_BBox(self, start, end, lms):
        min_x = 10000
        min_y = 10000
        max_x = -10000
        max_y = -10000

        for i in range(start, end):
            x = lms[i][0]
            y = lms[i][1]
            min_x = x if x < min_x else min_x
            min_y = y if y < min_y else min_y
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y

        return min_x, min_y, max_x, max_y
        # for tensor: xmin, xmax, ymin, ymax

    def mask_tensor(self, image):
        image_numpy = util.tensor2im(image.clone().detach())
        # util.save_image(image_numpy, '/home6/liuhy/contrastive-unpaired-translation/test/npmask.png' )
        image_cv2 = cv2.cvtColor(np.array(image_numpy), cv2.COLOR_BGR2RGB)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        # print(type(image_cv2), image_cv2.shape)

        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        align = F.FaceAlignment(landmarks_type=F.LandmarksType._3D, device=device)
        try:
            lms68 = align.get_landmarks(image_cv2)[0]
        except UserWarning:
            return []

        feature_id = [17, 22, 27, 36, 42, 48, 60, 68]
        feature_name = ['eyebrow1', 'eyebrow2', 'nose', 'eye1', 'eye2', 'lips', 'teeth']
        image_tensor = torch.ones_like(image)
        for i in range(len(feature_name)):
            xmin, xmax, ymin, ymax = self.return_BBox(feature_id[i], feature_id[i + 1], lms68)
            xmin, ymin, xmax, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
            # print(feature_name[i],': ',xmin, ymin, xmax, ymax)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    for j in range(3):
                        image_tensor[0][j][y][x] = 0

        # util.save_image(util.tensor2im(image_tensor * 255), '/home6/liuhy/contrastive-unpaired-translation/test/testmask3.png')
        # image_ret = image_tensor * image
        # image_numpy = util.tensor2im(image_ret)
        # util.save_image(image_numpy, '/home6/liuhy/contrastive-unpaired-translation/test/testmask2.png')
        # print("Current image saved...")

        return image_tensor

    def box2tensor(self, boxes, shape):
        image_tensor = torch.ones(shape)
        h = float(shape[2])
        w = float(shape[3])
        feature_name = ['eyebrow1', 'eyebrow2', 'nose', 'eye1', 'eye2', 'mouth']
        print(len(boxes))
        for i, box in enumerate(boxes):
            # for box in boxes:
            xmin, ymin, xmax, ymax = box
            # print(feature_name[i],': ',xmin, ymin, xmax, ymax)
            xmin, xmax = int(float(xmin[0]) * w / 1024.0), int(float(xmax[0]) * w / 1024.0)
            ymin, ymax = int(float(ymin[0]) * h / 1024.0), int(float(ymax[0]) * h / 1024.0)
            print(feature_name[i], ': ', xmin, ymin, xmax, ymax)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    for j in range(3):
                        image_tensor[0][j][y][x] = 0
        return image_tensor

    def calculate_masked_loss(self, A, B, bbox):
        masked_A = A * bbox
        masked_B = B * bbox
        npA = util.tensor2im(masked_A)
        npB = util.tensor2im(masked_B)
        util.save_image(npA, '/home6/liuhy/contrastive-unpaired-translation/test/npA.png')
        util.save_image(npB, '/home6/liuhy/contrastive-unpaired-translation/test/npB2.png')
        return torch.nn.L1Loss()(masked_A, masked_B)



# torch.Size([1, 64, 256, 256])
# torch.Size([1, 64, 256, 256])
# torch.Size([1, 256, 64, 64])
# torch.Size([1, 512, 32, 32])

# 两种方案
#   1. 每个src,tgt过vgg后四个feat_q&k，过4个netF(确认不同大小会不会对netF有影响）
#   2. 后两维upsample到512，cat到第一维，变成1，XXX
# L1：提人脸关键点（68）包围和, mask内部不算L1只算外部

# 68 keypoints through mtcnn / face alignment
# bbox of 5 guan both A&B, 求并集
# in result show the bounding box
# delete idt_B
