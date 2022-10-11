from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im


class InferenceDatasetWithOpposing(Dataset):

	def __init__(self, prepare_root, opposing_root, opts, transform=None):
		self.prepare_paths = sorted(data_utils.make_dataset(prepare_root))
		self.opposing_paths = sorted(data_utils.make_dataset(opposing_root))
		self.transform = transform
		self.opts = opts

		assert len(self.prepare_paths) == len(self.opposing_paths), \
			"Preparation teeth image list's length should be same to opposing ones"

	def __len__(self):
		return len(self.prepare_paths)

	def __getitem__(self, index):
		prepare_path = self.prepare_paths[index]
		prepare_im = Image.open(prepare_path)
		prepare_im = prepare_im.convert('L')

		opposing_path = self.opposing_paths[index]
		opposing_im = Image.open(opposing_path)
		opposing_im = opposing_im.convert('L')

		zero_im = np.expand_dims(np.zeros(prepare_im.size, dtype=np.uint8), axis=2)
		prepare_im = np.expand_dims(np.array(prepare_im), axis=2)
		opposing_im = np.expand_dims(np.array(opposing_im), axis=2)
		from_im = Image.fromarray(np.concatenate((prepare_im, opposing_im, zero_im), axis=2))

		if self.transform:
			from_im = self.transform(from_im)
		return from_im


class InferenceDatasetWithOpposingGap(Dataset):

	def __init__(self, prepare_root, opposing_root, gap_root, opts, transform=None):
		self.prepare_paths = sorted(data_utils.make_dataset(prepare_root))
		self.opposing_paths = sorted(data_utils.make_dataset(opposing_root))
		self.gap_paths = sorted(data_utils.make_dataset(gap_root))
		self.transform = transform
		self.opts = opts

		assert len(self.prepare_paths) == len(self.opposing_paths), \
			"Preparation teeth image list's length should be same to opposing ones"

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

		prepare_im = np.expand_dims(np.array(prepare_im), axis=2)
		opposing_im = np.expand_dims(np.array(opposing_im), axis=2)
		gap_im = np.expand_dims(np.array(gap_im), axis=2)
		from_im = Image.fromarray(np.concatenate((prepare_im, opposing_im, gap_im), axis=2))

		if self.transform:
			from_im = self.transform(from_im)
		return from_im
