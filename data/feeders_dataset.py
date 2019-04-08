import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class FeedersDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_goal = os.path.join(opt.dataroot, opt.phase + '_goal')  # create a path '/path/to/data/train_goal'
        self.goal_paths = sorted(make_dataset(self.dir_goal, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.goal_size = len(self.goal_paths)  # get the size of dataset A

        self.dir_feeders={}
        self.feeder_paths={}
        self.feeder_sizes={}
        for feeder in opt.feeders.split(','):
            self.dir_feeders[feeder] = os.path.join(opt.dataroot, opt.phase + feeder)  # create a path '/path/to/data/train_x'
            self.feeder_paths[feeder] = sorted(make_dataset(self.dir_feeders[feeder], opt.max_dataset_size))    # load images from '/path/to/data/trainB'
            self.feeder_sizes[feeder] = len(self.feeder_paths[feeder])  # get the size of dataset B
        input_nc=1
        output_nc=1 #I'm assuming grayscale everywhere
        params = None#{'crop_pos':?}
        self.transform = get_transform(self.opt, grayscale=True)
        #self.opt.preprocess+='_rot'
        self.transformFeeder= get_transform(self.opt, params, grayscale=True)
        #for feeder in opt.feeders.split(','):
        #    feeder == 'synthetic':
        #    t = get_transform(self.opt, grayscale=(output_nc == 1))
        #    self.transformFeeder[feeder]=t

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
        goal_path = self.goal_paths[index % self.goal_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            raise NotImplemented('Serial not implemented')
            index_B = index % self.B_size
        #else:   # randomize the index for domain B to avoid fixed pairs.
            feeder = random.choice(self.feeders)
        if feeder=='synthetic':
            feeder_path='sythesized'
            feeder_img = generate_synthetic(self.opt)
        else:
            index_feeder = random.randint(0, self.feeder_sizes[feeder] - 1)
            feeder_path = self.feeder_paths[feeder][index_feeder]
            feeder_img = Image.open(feeder_path).convert('RGB')
        goal_img = Image.open(goal_path).convert('RGB')
        # apply image transformation
        A = self.transform(goal_img)
        B = self.transformFeeder(feeder_img)

        return {'A': A, 'B': B, 'A_paths': goal_path, 'B_paths': feeder_path, 'B_branch':feeder}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.goal_size, self.B_size)
