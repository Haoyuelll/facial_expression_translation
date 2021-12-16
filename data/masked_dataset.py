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
from PIL import Image
import random
import util.util as util
import json
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as transforms
import face_alignment as F
from data.base_dataset import BaseDataset, get_transform
# import torchvision.transforms as transforms
from data.image_folder import make_dataset


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
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
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
        self.device = f'cuda:{self.opt.gpu_ids[0]}' if self.opt.gpu_ids else 'cpu'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_masks = self.mask_face('A', opt.dataroot, self.A_paths)  # get the bounding box for images in dataset A
        self.B_masks = self.mask_face('B', opt.dataroot, self.B_paths)  # get the bounding box for images in dataset A

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
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_box = self.B_masks[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        scale_size = 256
        img2tensor = transforms.Compose([transforms.Resize([scale_size, scale_size], Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        A = img2tensor(A_img)
        B = img2tensor(B_img)
        A_box = [int(float(i) / A_img.width * scale_size) for i in A_box]
        B_box = [int(float(i) / B_img.width * scale_size) for i in B_box]
        A_box, B_box = str(A_box), str(B_box)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_box': A_box, 'B_box': B_box}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def return_BBox(self, lms, start, end):
        """ Return the bounding box of the input Xs and Ys, start_id=start, end_id=end 
        """
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
        min_x = 0 if min_x < 0 else min_x
        min_y = 0 if min_x < 0 else min_y
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        return [min_x, min_y, max_x, max_y]

    def calc_center(self, lms, start=0, end=17):
        """ Calculate the center of each face in the pic, start_id=start, end_id=end 
            default start at 0 and end at 17 (the face)
        """
        sum_x, sum_y = 0, 0
        for i in range(start, end):
            sum_x += lms[i][0]
            sum_y += lms[i][1]

        ave_x = sum_x / (end - start)
        ave_y = sum_y / (end - start)

        return ave_x, ave_y

    def mask_face(self, domain, rootdir, paths):
        """ Saving bouding box of the major face in each pic as a json file in {rootdir}/train{domain}/boxes.json.

        Parameters: 
            rootdir: directory of the training dataset
            domain: 'A' or 'B', to find or create corresponding lms directory and json file
            paths: path to the sorted training dataset
            Parser: --mask_start & --mask_end: the starting and ending id for bbox 

        step1: try to load the bbox dict (key: pic name; val: x_left, x_right, y_left, y_right of the bbox)from json file and create new if file does not exist.
        step2: load the 4 points of bbox for each pic, if error try to load the landmarks from .npy file and write the box into json file, if error again, calculate the landmarks with face_alignment

        Returns:
            boxes: list of 4 points of each pic's bbox

        """
        boxes = []
        size = len(paths)
        img = Image.open(paths[0])
        image_w = img.width
        if self.opt.mask_start == 0 and self.opt.mask_end == 27:
            mode = '1'
        elif self.opt.mask_start == 17 and self.opt.mask_end == 68:
            mode = '2'
        else:
            mode = '3'

        if not os.path.isfile(f'{rootdir}/train{domain}/bbox_{mode}.json'):
            os.mknod(f'{rootdir}/train{domain}/bbox_{mode}.json')

        with open(f'{rootdir}/train{domain}/bbox_{mode}.json', 'r+') as box_file:
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
                    # raise KeyError
                    bbox = box_dict[name]
                except KeyError:
                    lms68 = []
                    try:
                        lms68 = np.load(f'{rootdir}/train{domain}_lms/{name[:-4]}.npy')
                        x, y = self.calc_center(0, 17, lms68)
                        if x < image_w / 4 or x > image_w / 4 * 3:
                            raise UserWarning
                    except:
                        if i % 100 == 0:
                            print(f'Calculating landmarks  {i}/{size}')
                        if not os.path.isdir(f'{rootdir}/train{domain}_lms'):
                            os.mkdir(f'{rootdir}/train{domain}_lms')
                        self.device = f'cuda:{self.opt.gpu_ids[0]}' if self.opt.gpu_ids else 'cpu'
                        align = F.FaceAlignment(landmarks_type=F.LandmarksType._3D, device=self.device)
                        image = Image.open(path)
                        image_w = image.width
                        lms = align.get_landmarks(path)
                        if len(lms) > 1:
                            # if more than 1 face detected in the pic, only landmarks for the one in the center will be saved
                            for j in range(len(lms)):
                                x, y = self.calc_center(lms[j])
                                lms68 = lms[j]
                                if x > image_w / 4 and x < image_w / 4 * 3:
                                    break
                        np.save(f'{rootdir}/train{domain}_lms/{name[:-4]}.npy', lms68)
                    bbox = self.return_BBox(lms68, self.opt.mask_start, self.opt.mask_end)
                    box_dict[name] = bbox
                boxes.append(bbox)
            json.dump(box_dict, box_file)
        return boxes
