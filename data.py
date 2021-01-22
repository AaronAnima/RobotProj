import argparse
import numpy as np
import torch
import random
import os
import gym
import cv2
import math
import torch
from collections import namedtuple, deque
from torchvision import transforms as T
from PIL import Image
from torchvision.utils import save_image, make_grid
from utils import string2bool


Data = namedtuple('data', ('image', 'label', 'img_name'))


class Args:
	def __init__(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--exp_name", default='exp_test', type=str)
		parser.add_argument("--lr", default=0.0002, type=float)
		parser.add_argument("--epoch_num", default=1000, type=int)
		parser.add_argument("--batchsize", default=20, type=int)
		parser.add_argument("--input_h", default=60, type=int)
		parser.add_argument("--input_w", default=80, type=int)
		parser.add_argument("--order", default=3, type=int)
		parser.add_argument("--is_grey", default='True', type=string2bool)
		
		self.parser = parser.parse_args()
		
		
def pre_pro_img(img, img_h, img_w, device):
	img = img.transpose((2, 0, 1))
	img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
	img_ts = torch.from_numpy(img)
	resize = T.Compose([T.ToPILImage(),
						T.Resize((img_h, img_w), interpolation=Image.CUBIC),
						T.ToTensor()])
	img_ts = resize(img_ts)
	# [0, 1] -> [-1, 1]
	img_ts = img_ts*2 - 1
	return img_ts.unsqueeze(0).to(device)


def key_for_sort(item):
	return item.img_name
	

class DataLoader(object):
	def __init__(self, data_path, device, batchsize, shape, order):
		self.data_path = data_path
		self.device = device
		self.paired_data = []
		self.batches = []
		self.batchsize = batchsize
		self.img_h, self.img_w = shape
		self.order = order
		
		# load label first
		label_dict = {}
		txt_file = open(data_path + '/steer_list.txt')
		for line in txt_file.readlines():
			line = line.strip()
			content = line.split(' ')
			label_ts = torch.FloatTensor([float(content[1])]).to(device).unsqueeze(0)
			label_dict[content[0]] = label_ts
		t_paired_data = []
		# then load image and make pair
		for img_name in os.listdir(self.data_path + '/cw-color/'):
			label = label_dict[img_name]
			img = cv2.imread(os.path.join(self.data_path + '/cw-color/', img_name), cv2.IMREAD_GRAYSCALE)
			# cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE
			img = np.expand_dims(img, 2)
			img = pre_pro_img(img, self.img_h, self.img_w, self.device)
			t_paired_data.append(Data(image=img, label=label, img_name=img_name))
		
		# process order
		t_paired_data.sort(key=key_for_sort)
		my_queue = deque(maxlen=self.order)
		for idx in range(self.order):
			my_queue.append(t_paired_data[idx].image)
		for idx, item in enumerate(t_paired_data):
			my_queue.append(item.image)
			cur_obs = torch.cat([img for img in my_queue], dim=1)
			self.paired_data.append(Data(image=cur_obs, label=item.label, img_name=item.img_name))
		
		# setup pointer
		self.p = 0
		self.length = math.ceil(len(self.paired_data)*1.0 / self.batchsize)
		# cut_into_batches
		for idx in range(self.length):
			# setup start and end idx
			start_idx = idx * self.batchsize
			end_idx = (idx + 1) * self.batchsize
			if end_idx + 1 > len(self.paired_data):
				end_idx = len(self.paired_data) - 1
			# assert data.image.shape == [1, c, h, w]
			batch_img = torch.cat([data.image for data in self.paired_data[start_idx:end_idx]], dim=0)
			# assert data.image.shape == [1, 1]
			batch_label = torch.cat([data.label for data in self.paired_data[start_idx:end_idx]], dim=0)
			# add batch tuple to batches
			self.batches.append((batch_img, batch_label))
			
		self.whole_img = torch.cat([data.image for data in self.paired_data], dim=0)
		self.whole_steer = torch.cat([data.label for data in self.paired_data], dim=0)
		txt_file.close()
		
	def load(self):
		# load batch
		ret_batch = self.batches[self.p]
		
		# update pointer
		self.p = (self.p + 1) % self.length
		
		# return
		return ret_batch
	
	def load_whole(self):
		return self.whole_img, self.whole_steer
	
	def eval(self, policy, log_path, name):
		with torch.no_grad():
			output = open(log_path + name + '.txt', mode='w', buffering=1)
			for data in self.paired_data:
				out_steer = policy(data.image)
				content = data.img_name + '\t' + str(round(out_steer.cpu().numpy().item(), 2)) + '\n'
				output.write(content)
			output.close()
			
	def __len__(self):
		return self.length

