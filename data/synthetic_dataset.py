import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from synthetic_text_gen import SyntheticText, apply_tensmeyer_brightness, grid_distortion
#from PIL import Image
import cv2, json
import random
import torch
#from util import grid_distortion


class SyntheticDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        parser.add_argument('--scale_height', type=int, default=32, help='scale images to this height')
        parser.add_argument('--text_only', type=bool, default=False, help='only use text images')
        return parser


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if opt.text_only:
            self.dir_A = os.path.join(opt.dataroot, opt.phase,'text')  # create a path '/path/to/data/train'
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/train'
        self.augment = opt.phase=='train'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        btoA = self.opt.direction == 'BtoA'
        
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        #self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.scale_height = self.opt.scale_height
        
        if opt.syn_file is None:
            self.use_warp=-1
            self.textGen = [SyntheticText('../data/fonts/text_fonts','../data/OANC_text',line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=2.5,text_len=35, use_warp=0.4,warp_std=[1,1.4])]
            
            if not opt.text_only:
                self.textGen.append(SyntheticText('../data/fonts/handwritten_fonts','../data/OANC_text',line_prob=0.85,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=2.5,text_len=35, use_warp=0.6,warp_std=[1,2]))
        else:
            with open(opt.syn_file) as f:
                syn_params = json.load(f)
            self.textGen=[]
            for gen in syn_params:
                self.textGen.append(SyntheticText(
                    gen['font_dir'],
                    gen['text_dir'],
                    line_prob=gen['line_prob'],
                    line_thickness=gen['line_thickness'],
                    line_var=gen['line_var'],
                    pad=gen['pad'],
                    gaus_noise=gen['gaus_noise'],
                    hole_prob=gen['hole_prob'],
                    hole_size=gen['hole_size'],
                    neighbor_gap_var=gen['neighbor_gap_var'],
                    rot=gen['rot'],
                    text_len=gen['text_len'],
                    use_warp=gen['use_warp'],
                    warp_std=gen['warp_std'],
                    warp_intr=gen['warp_intr']
                    ))
            self.use_warp = gen['use_warp']
            

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_img = cv2.imread(A_path,0) #Image.open(A_path).convert('RGB')
        if A_img is None or A_img.shape[1]/A_img.shape[0]<0.85:
            return self[(index+100)%len(self)]
        if self.augment:
            A_img = apply_tensmeyer_brightness(A_img,20)
        gen = random.choice(self.textGen)
        B_img,text = gen.getSample()
        if B_img.shape[1]/B_img.shape[0]<0.85:
            return self[(index+100)%len(self)]
        # apply image transformation
        #A = self.transform_A(A_img)
        #B = self.transform_B(B_img)
    
        A_w = round(A_img.shape[1] * float(self.scale_height)/A_img.shape[0])
        A_img = cv2.resize(A_img,(A_w,self.scale_height),interpolation = cv2.INTER_CUBIC)
        B_w = round(B_img.shape[1] * float(self.scale_height)/B_img.shape[0])
        B_img = cv2.resize(B_img,(B_w,self.scale_height),interpolation = cv2.INTER_CUBIC)

        assert(A_img.size>0 and B_img.size>0)

        #cv2.imwrite('test/A{}.png'.format(index),A_img)
        #cv2.imwrite('test/B{}.png'.format(index),255*B_img)
        A_img = 1-(A_img/128.0)
        B_img = 2.0*B_img-1.0
        if self.augment:
            if random.random() < self.use_warp:
                A_img = grid_distortion.warp_image(A_img) #use default params of start-follow-read

        A = torch.from_numpy(A_img)[None,...].float()
        B = torch.from_numpy(B_img)[None,...].float()
        assert(A.size(2)>0 and B.size(2)>0)
        #print('A:{}, B:{}'.format(A.size(),B.size()))

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': 'synthetic', 'B_text':text}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size
