import cv2
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
import numpy as np


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im


class ImagesDatasetToothInpainting(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, flip=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.flip = flip

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')

        if self.flip:
            if np.random.randint(2):
                from_im = from_im.transpose(method=Image.FLIP_LEFT_RIGHT)
                to_im = to_im.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return from_im, to_im


class ImagesDatasetWithOpposing(Dataset):

    def __init__(self, prepare_root, opposing_root, object_root, opts, target_transform=None, source_transform=None, flip=None):
        self.prepare_paths = sorted(data_utils.make_dataset(prepare_root))
        self.opposing_paths = sorted(data_utils.make_dataset(opposing_root))
        self.object_paths = sorted(data_utils.make_dataset(object_root))

        assert len(self.prepare_paths) == len(self.opposing_paths), \
            "Preparation teeth image list's length should be same to opposing ones"

        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.flip = flip

    def __len__(self):
        return len(self.prepare_paths)

    def __getitem__(self, index):
        prepare_path = self.prepare_paths[index]
        prepare_im = Image.open(prepare_path)
        prepare_im = prepare_im.convert('L')

        opposing_path = self.opposing_paths[index]
        opposing_im = Image.open(opposing_path)
        opposing_im = opposing_im.convert('L')

        object_path = self.object_paths[index]
        to_im = Image.open(object_path).convert('RGB')

        assert os.path.basename(prepare_path) == os.path.basename(opposing_path) == os.path.basename(object_path), \
            "Image name should be same"

        zero_im = np.expand_dims(np.zeros(prepare_im.size, dtype=np.uint8), axis=2)
        prepare_im = np.expand_dims(np.array(prepare_im), axis=2)
        opposing_im = np.expand_dims(np.array(opposing_im), axis=2)
        from_im = Image.fromarray(np.concatenate((prepare_im, opposing_im, zero_im), axis=2))

        if self.flip:
            if np.random.randint(2):
                from_im = from_im.transpose(method=Image.FLIP_LEFT_RIGHT)
                to_im = to_im.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return from_im, to_im


class ImagesDatasetToothInpaintingWoCenter(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, flip=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.flip = flip

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        # The value of image's center is replaced with zero
        w, h = from_im.size
        lt, ut = w // 2 - w // 2 // 3, w // 2 + w // 2 // 4
        from_im_np = np.asarray(from_im, dtype=np.uint8)
        from_im_np[lt:ut, lt:ut, :] = 0
        from_im = Image.fromarray(from_im_np)

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')

        if self.flip:
            if np.random.randint(2):
                from_im = from_im.transpose(method=Image.FLIP_LEFT_RIGHT)
                to_im = to_im.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return from_im, to_im


class ImagesDatasetWithOpposingGap(Dataset):

    def __init__(self, prepare_root, opposing_root, gap_root, object_root, opts, target_transform=None, source_transform=None, flip=None):
        self.prepare_paths = sorted(data_utils.make_dataset(prepare_root))
        self.opposing_paths = sorted(data_utils.make_dataset(opposing_root))
        self.gap_paths = sorted(data_utils.make_dataset(gap_root))
        self.object_paths = sorted(data_utils.make_dataset(object_root))

        assert len(self.prepare_paths) == len(self.opposing_paths), \
            "Preparation teeth image list's length should be same to opposing ones"

        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.flip = flip

    def __len__(self):
        return len(self.prepare_paths)

    def __getitem__(self, index):
        prepare_path = self.prepare_paths[index]
        prepare_im = Image.open(prepare_path)
        prepare_im = prepare_im.convert('L')

        opposing_path = self.opposing_paths[index]
        opposing_im = Image.open(opposing_path)
        opposing_im = opposing_im.convert('L')

        gap_path = self.gap_paths[index]
        gap_im = Image.open(gap_path)
        gap_im = gap_im.convert('L')

        object_path = self.object_paths[index]
        to_im = Image.open(object_path).convert('RGB')

        assert prepare_path.split('_')[0] == opposing_path.split('_')[0] == gap_path.split('_')[0] == object_path.split('_')[0], \
            "Image name should be same"

        prepare_im = np.expand_dims(np.array(prepare_im), axis=2)
        opposing_im = np.expand_dims(np.array(opposing_im), axis=2)
        gap_im = np.expand_dims(np.array(gap_im), axis=2)
        from_im = Image.fromarray(np.concatenate((prepare_im, opposing_im, gap_im), axis=2))

        if self.flip:
            if np.random.randint(2):
                from_im = from_im.transpose(method=Image.FLIP_LEFT_RIGHT)
                to_im = to_im.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return from_im, to_im