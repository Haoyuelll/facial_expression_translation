"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path

from torch._C import device
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torch
import face_alignment as F
import json
import numpy as np
# from data.image_folder import make_dataset
from PIL import Image


class MaskedDataset(BaseDataset):
    """
    Customized dataset class can load unaligned/unpaired datasets with masks.

    It requires 4 directories to host training images from domain A '/path/to/data/trainA'
    domain B '/path/to/data/trainB', mask_domain A /path/to/data/maskA, and mask_domain B /path/to/data/maskB.

    Train the model with the dataset flag '--dataroot /path/to/data --data_mode masked'.
    In test phase, directories of:
    '/path/to/data/testA', '/path/to/data/testB', '/path/to/data/maskA', and '/path/to/data/maskB'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--mask_start', type=int, default=0, help='0 to 68, the start id for key points caught in face_alignment to calculate bounding box')
        parser.add_argument('--mask_end', type=int, default=27, help='0 to 68, the start id for key points caught in face_alignment to calculate bounding box')
        # parser.set_defaults(max_dataset_size=100, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.d

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)d
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_masks = self.mask_face('A', opt.dataroot, self.A_paths)  # get the bounding box for images in dataset A
        self.B_masks = self.mask_face('B', opt.dataroot, self.B_paths)  # get the bounding box for images in dataset B
        # self.A_masks = self.mask_box('A', opt.dataroot, self.A_paths)  # get the bounding box for images in dataset A
        # self.B_masks = self.mask_box('B', opt.dataroot, self.B_paths)  # get the bounding box for images in dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_box = self.A_masks[index % self.A_size]
        # print('in __getitem__(): A_mask = ', A_mask)
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_box = self.B_masks[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        scale_size = self.opt.crop_size if is_finetuning else self.opt.load_size
        A_mask = self.box2tensor(A_box, (1, 3, scale_size, scale_size), A_img.width, A_img.height)
        A_mask_img = util.tensor2im(A_mask)
        B_mask = self.box2tensor(B_box, (1, 3, scale_size, scale_size), B_img.width, B_img.height)
        B_mask_img = util.tensor2im(B_mask)

        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = self.get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        A_mask_img = transform(Image.fromarray(A_mask_img))
        A_mask = self.tensorcvt(A_mask_img, scale_size)
        B_mask_img = transform(Image.fromarray(B_mask_img))
        B_mask = self.tensorcvt(B_mask_img, scale_size)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

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

        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        return [min_x, min_y, max_x, max_y]

    def calc_center(self, start, end, lms):
        sum_x, sum_y = 0, 0
        for i in range(start, end):
            sum_x += lms[i][0]
            sum_y += lms[i][1]

        ave_x = sum_x / (end - start)
        ave_y = sum_y / (end - start)

        return ave_x, ave_y

    def mask_box(self, domain, rootdir, paths):
        boxes = []
        size = len(paths)
        with open(f'{rootdir}/train{domain}/masks.json', 'r+') as mask_file:
            try:
                masks_dict = json.load(mask_file)
                # raise json.decoder.JSONDecodeError('msg', 'msd', 2)
            except json.decoder.JSONDecodeError:
                masks_dict = {}
                print('No existing masks for train' + domain)
            mask_file.seek(0)

            for i, path in enumerate(paths):
                name = path[len(rootdir) + 8:]

                try:    # use mask from the json file if exists
                    box = self.box_all(masks_dict[name]) if len(masks_dict[name]) > 1 else masks_dict[name]    # adding a bouding box for all the features
                    boxes.append(box)
                    masks_dict[name] = box
                    if i % 100 == 0:
                        print(f'Loading masks  {i}/{size}')

                except KeyError:    # calculate if mask for the specific pic does not exist
                    if i % 100 == 0:
                        print(f'Calculating masks  {i}/{size}')
                    device = f'cuda:{self.opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
                    align = F.FaceAlignment(landmarks_type=F.LandmarksType._3D, device=device)
                    image = Image.open(path)
                    image_w = image.width
                    lms68 = []
                    try:
                        lms = align.get_landmarks(path)
                        for i in range(len(lms)):
                            x, y = self.calc_center(0, 17, lms[i])
                            lms68 = lms[i]
                            if x > image_w / 4 and x < image_w / 4 * 3:
                                break
                    except UserWarning:
                        return []
                    feature_id = [17, 22, 27, 36, 42, 48, 68]
                    feature_name = ['eyebrow1', 'eyebrow2', 'nose', 'eye1', 'eye2', 'mouth']
                    bbox = []
                    for i in range(len(feature_name)):
                        bbox.append(self.return_BBox(feature_id[i], feature_id[i + 1], lms68))
                    box = self.box_all(bbox)    # adding a bouding box for all the features
                    boxes.append(box)
                    masks_dict[name] = box
            json.dump(masks_dict, mask_file)
        if len(boxes) > 200:
            with open('/home6/liuhy/contrastive-unpaired-translation/test/boxes.txt', 'w') as f:
                f.write(str(boxes))
        return boxes

    def mask_face(self, domain, rootdir, paths):
        boxes = []
        size = len(paths)

        if not os.path.isfile(f'{rootdir}/train{domain}/boxes.json'):
            os.mknod(f'{rootdir}/train{domain}/boxes.json')

        with open(f'{rootdir}/train{domain}/boxes.json', 'r+') as box_file:
            try:
                box_dict = json.load(box_file)
                # raise json.decoder.JSONDecodeError('msg', 'msd', 2)
            except json.decoder.JSONDecodeError:
                box_dict = {}
                print('No existing masks for train' + domain)
            box_file.seek(0)

            for i, path in enumerate(paths):
                if i % 100 == 0:
                    print(f'Loading masks  {i}/{size}')
                name = path[len(rootdir) + 8:]
                try:
                    bbox = box_dict[name]
                except KeyError:
                    lms68 = []
                    try:
                        lms68 = np.load(f'{rootdir}/train{domain}_lms/{name[:-4]}.npy')
                    except:
                        if not os.path.isdir(f'{rootdir}/train{domain}_lms'):
                            os.mkdir(f'{rootdir}/train{domain}_lms')
                        device = f'cuda:{self.opt.gpu_ids}' if torch.cuda.is_available() else 'cpu'
                        align = F.FaceAlignment(landmarks_type=F.LandmarksType._3D, device=device)
                        image = Image.open(path)
                        image_w = image.width
                        lms = align.get_landmarks(path)
                        for i in range(len(lms)):
                            x, y = self.calc_center(0, 17, lms[i])
                            lms68 = lms[i]
                            if x > image_w / 4 and x < image_w / 4 * 3:
                                break
                        np.save(f'{rootdir}/train{domain}_lms/{name[:-4]}.npy', lms68)
                    bbox = self.return_BBox(self.opt.mask_start, self.opt.mask_end, lms68)
                    box_dict[name] = bbox
                boxes.append(bbox)
            json.dump(box_dict, box_file)
        return boxes

    def box2tensor(self, box, shape, w, h):
        image_tensor = torch.ones(shape) * 255
        # feature_name = ['eyebrow1', 'eyebrow2', 'nose', 'eye1', 'eye2', 'mouth']
        xmin, ymin, xmax, ymax = box
        xmin = int(float(xmin) * shape[2] / w)
        xmax = int(float(xmax) * shape[2] / w)
        ymin = int(float(ymin) * shape[2] / h)
        ymax = int(float(ymax) * shape[2] / h)
        if xmax >= 286 or ymax >= 286:
            print(shape[2], w, h, shape[2] / w)
            print(box)
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                for j in range(3):
                    image_tensor[0][j][y][x] = 0
        return image_tensor

    def tensorcvt(self, tensor, w):
        for x in range(w):
            for y in range(w):
                for c in range(3):
                    tensor[c][x][y] = 1 if abs(tensor[c][x][y].item() - 1) < 10e-5 else 0
        # tensor = torch.unsqueeze(tensor, 0)
        # tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
        return tensor

    def box_all(self, boxes):
        xmin, ymin, xmax, ymax = boxes[0]
        for i in range(1, len(boxes)):
            xmin = boxes[i][0] if boxes[i][0] < xmin else xmin
            ymin = boxes[i][1] if boxes[i][1] < ymin else ymin
            xmax = boxes[i][2] if boxes[i][2] > xmax else xmax
            ymax = boxes[i][3] if boxes[i][3] > ymax else ymax
        return [xmin, ymin, xmax, ymax]

    def get_transform(self, opt, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
        transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base=4, method=method)))
        transform_list.append(transforms.Lambda(lambda img: self.__flip(img)))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __make_power_2(self, img, base, method=Image.BICUBIC):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if h == oh and w == ow:
            return img

        return img.resize((w, h), method)

    def __flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
