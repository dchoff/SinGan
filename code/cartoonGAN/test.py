import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage import io

import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = 'test_img')
parser.add_argument('--load_size', default = 250)
parser.add_argument('--model_path', default = './cartoonGAN/pretrained_model')
parser.add_argument('--style', default = None)
parser.add_argument('--output_dir', default = 'test_output')
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--denoised', type=bool, default=False)
parser.add_argument('--keep_denoised', type=bool, default=False)
parser.add_argument('--mod_name', default = None)

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)
if not os.path.exists(opt.output_dir+'/results'): os.mkdir(opt.output_dir+'/results')
if opt.mod_name is not None:
	if not os.path.exists(opt.output_dir+'/results/'+opt.mod_name): os.mkdir(opt.output_dir+'/results/'+opt.mod_name)

if not os.path.exists(opt.output_dir+'/denoised_images'): os.mkdir(opt.output_dir+'/denoised_images')
styles = ['Hayao', 'Hosoda', 'Paprika', 'Shinkai']

if opt.style is None:
	full_output = opt.output_dir + '/results'
	if opt.mod_name is not None:
		if not os.path.exists(opt.output_dir+'/results/'+opt.mod_name): os.mkdir(opt.output_dir+'/results/'+opt.mod_name)
		full_output = opt.output_dir + '/results/' + opt.mod_name

	for st in styles:
		if not os.path.exists(full_output+'/'+st): os.mkdir(full_output+'/'+st)
		# load pretrained model
		model = Transformer()
		model.load_state_dict(torch.load(os.path.join(opt.model_path, st + '_net_G_float.pth')))
		model.eval()

		io.use_plugin('pil')

		if opt.gpu > -1:
			print('GPU mode')
			model.cuda()
		else:
			print('CPU mode')
			model.float()


		# python test.py --input_dir ../code/results/SinGAN_2020-02-20_10-57-01 --style Hosoda --gpu 0
		# Denoise SinGAN results first
		print('\n')	
		if not opt.denoised:
			numFiles = len(os.listdir(opt.input_dir))
			i = 0
			for files in os.listdir(opt.input_dir):
				print('Denoising image {} of {}\r'.format(i, numFiles), end='', flush=True)

				ext = os.path.splitext(files)[1]
				if ext not in valid_ext:
					continue

				curr_img = img_as_float(io.imread(os.path.join(opt.input_dir, files)))
				# Estimate noise variance in image
				sigma_est = np.mean(estimate_sigma(curr_img, multichannel=True))

				patch_kw = dict(patch_size=5,      # 5x5 patches
				                patch_distance=6,  # 13x13 search area
				                multichannel=True)

				denoise = denoise_nl_means(curr_img, h=1.15 * sigma_est, fast_mode=False, **patch_kw)
				io.imsave(os.path.join(opt.output_dir+'/denoised_images', files)+'.jpg', denoise)
				i+=1

		opt.denoised = True

		print('Processing denoised images for style: {}'.format(st))
		i = 0
		numFiles = len(os.listdir(opt.output_dir+'/denoised_images'))
		for files in os.listdir(opt.output_dir+'/denoised_images'):
			print('Image {} of {}\r'.format(i, numFiles), end='', flush=True)

			ext = os.path.splitext(files)[1]
			if ext not in valid_ext:
				continue
			# load image
			input_image = Image.open(os.path.join(opt.output_dir+'/denoised_images', files)).convert("RGB")
			# resize image, keep aspect ratio
			h = input_image.size[0]
			w = input_image.size[1]
			ratio = h *1.0 / w
			if ratio > 1:
				h = opt.load_size
				w = int(h*1.0/ratio)
			else:
				w = opt.load_size
				h = int(w * ratio)
			input_image = input_image.resize((h, w), Image.BICUBIC)
			input_image = np.asarray(input_image)
			# RGB -> BGR
			input_image = input_image[:, :, [2, 1, 0]]
			input_image = transforms.ToTensor()(input_image).unsqueeze(0)
			# preprocess, (-1, 1)
			input_image = -1 + 2 * input_image 
			if opt.gpu > -1:
				input_image = Variable(input_image, volatile=True).cuda()
			else:
				input_image = Variable(input_image, volatile=True).float()
			# forward
			output_image = model(input_image)
			output_image = output_image[0]
			# BGR -> RGB
			output_image = output_image[[2, 1, 0], :, :]
			# deprocess, (0, 1)
			output_image = output_image.data.cpu().float() * 0.5 + 0.5
			# save
			vutils.save_image(output_image, os.path.join(full_output+'/'+st, files[:-4] + '_' + st + '.jpg'))
			i += 1

else:
	if not os.path.exists(opt.output_dir+'/results'): os.mkdir(opt.output_dir+'/results')
	full_output = opt.output_dir + '/results'
	if opt.mod_name is not None:
		if not os.path.exists(opt.output_dir+'/results/'+opt.mod_name): os.mkdir(opt.output_dir+'/results/'+opt.mod_name)
		full_output = opt.output_dir + '/results/' + opt.mod_name	
	# load pretrained model
	model = Transformer()
	model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
	model.eval()

	io.use_plugin('pil')

	if opt.gpu > -1:
		print('GPU mode')
		model.cuda()
	else:
		print('CPU mode')
		model.float()


	# python test.py --input_dir ../code/results/SinGAN_2020-02-20_10-57-01 --style Hosoda --gpu 0
	# Denoise SinGAN results first
	if not opt.denoised:
		for files in os.listdir(opt.input_dir):
			ext = os.path.splitext(files)[1]
			if ext not in valid_ext:
				continue

			curr_img = img_as_float(io.imread(os.path.join(opt.input_dir, files)))
			# Estimate noise variance in image
			sigma_est = np.mean(estimate_sigma(curr_img, multichannel=True))

			patch_kw = dict(patch_size=5,      # 5x5 patches
			                patch_distance=6,  # 13x13 search area
			                multichannel=True)

			denoise = denoise_nl_means(curr_img, h=1.15 * sigma_est, fast_mode=False, **patch_kw)
			io.imsave(os.path.join(opt.output_dir+'/denoised_images', files)+'.jpg', denoise)


	for files in os.listdir(opt.output_dir+'/denoised_images'):
		ext = os.path.splitext(files)[1]
		if ext not in valid_ext:
			continue
		# load image
		input_image = Image.open(os.path.join(opt.output_dir+'/denoised_images', files)).convert("RGB")
		# resize image, keep aspect ratio
		h = input_image.size[0]
		w = input_image.size[1]
		ratio = h *1.0 / w
		if ratio > 1:
			h = opt.load_size
			w = int(h*1.0/ratio)
		else:
			w = opt.load_size
			h = int(w * ratio)
		input_image = input_image.resize((h, w), Image.BICUBIC)
		input_image = np.asarray(input_image)
		# RGB -> BGR
		input_image = input_image[:, :, [2, 1, 0]]
		input_image = transforms.ToTensor()(input_image).unsqueeze(0)
		# preprocess, (-1, 1)
		input_image = -1 + 2 * input_image 
		if opt.gpu > -1:
			input_image = Variable(input_image, volatile=True).cuda()
		else:
			input_image = Variable(input_image, volatile=True).float()
		# forward
		output_image = model(input_image)
		output_image = output_image[0]
		# BGR -> RGB
		output_image = output_image[[2, 1, 0], :, :]
		# deprocess, (0, 1)
		output_image = output_image.data.cpu().float() * 0.5 + 0.5
		# save
		vutils.save_image(output_image, os.path.join(full_output, files[:-4] + '_' + opt.style + '.jpg'))			

if os.path.exists(opt.output_dir+'/denoised_images') and not opt.keep_denoised: shutil.rmtree(opt.output_dir+'/denoised_images')
			

print('Done!')
