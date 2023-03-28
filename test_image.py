# test phase
import torch
from torch.autograd import Variable
from net import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
from homography_warp import homography_warp


def load_model(path, input_nc, output_nc):

	nest_model = DenseFuse_net(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))
    
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
	# encoder
	en_r = model.encoder1(img1)
	en_v = model.encoder2(img2)
	
	# fusion
	f = model.fusion(en_r, en_v, strategy_type=strategy_type)
	
	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, unexp_path, ovexp_path, output_path_root, index, fusion_type, network_type, strategy_type, model_type, ssim_weight_str, mode, set_align):    
	start = time.time()
	if mode == 'LAB':
		if set_align:
			un_img, ov_img, a1, a2, b1, b2 = homography_warp(unexp_path, ovexp_path)
		else:
			un_img, a1, b1 = utils.get_test_images(unexp_path, height=None, width=None, mode=mode)
			ov_img, a2, b2 = utils.get_test_images(ovexp_path, height=None, width=None, mode=mode)
	elif mode == 'L':
		un_img = utils.get_test_images(unexp_path, height=None, width=None, mode=mode)
		ov_img = utils.get_test_images(ovexp_path, height=None, width=None, mode=mode)
        
	if args.cuda:
		un_img = un_img.cuda()
		ov_img = ov_img.cuda()
	un_img = Variable(un_img, requires_grad=False)
	ov_img = Variable(ov_img, requires_grad=False)
	dimension = un_img.size()
    
    
	img_fusion = _generate_fusion_image(model, strategy_type, un_img, ov_img)
    
	############################ multi outputs ##############################################
	file_name = str(index) + model_type + '.jpg'
	output_path = output_path_root + file_name

	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
                                
	img = img.transpose(1, 2, 0).astype('uint8')
    
	if mode == 'LAB':
		utils.save_images_lab(output_path, img, a1, a2, b1, b2)
	else:
		utils.save_images(output_path, img)
        
	print("elapsed time:",round(time.time()-start,4))
	print(output_path)


def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():
	# run demo
	# test_path = "images/test-RGB/"
	test_path = "images/"
	network_type = 'densefuse'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask
	model_type = '_HyperP_MB'
	set_align = True	# if you don't want to use homography warp to align source images, set this value to 'False'

	output_path = './outputs/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'LAB'
		model_path = args.model_path_gray

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[3])
		ssim_weight_str = args.ssim_path[3]
		model = load_model(model_path, in_c, out_c)
		for i in range(2,3):
			index = i
			unexp_path = test_path + str(index) + '_UN' + '.jpg'
			ovexp_path = test_path + str(index) + '_OV' + '.jpg'
			run_demo(model, unexp_path, ovexp_path, output_path, index, fusion_type, network_type, strategy_type, model_type, ssim_weight_str, mode, set_align)
	print('Done......')

if __name__ == '__main__':
	main()
