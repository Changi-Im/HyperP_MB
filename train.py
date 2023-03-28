# Training DenseFuse network
# auto-encoder
# pixel + ssim
# used dynamic lambda on total loss

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim
import cv2
from blur import generate_mask
import AGC

def main():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	source1_imgs_path = utils.list_images(args.dataset1)
	source2_imgs_path = utils.list_images(args.dataset2)
	target_imgs_path = utils.list_images(args.datasett)

	train_num = 460
	source1_imgs_path = source1_imgs_path[:train_num]
	source2_imgs_path = source2_imgs_path[:train_num]
	target_imgs_path = target_imgs_path[:train_num]

	data1 = np.array(source1_imgs_path)
	data2 = np.array(source2_imgs_path)
	data3 = np.array(target_imgs_path)

	s = np.arange(data1.shape[0])
	np.random.shuffle(s)
	source1_imgs_path = data1[s]
	source2_imgs_path = data2[s]          
	target_imgs_path = data3[s]

	# for i in range(5):
	i = 3
	train(i, source1_imgs_path, source2_imgs_path, target_imgs_path)


def train(i, source1_imgs_path, source2_imgs_path, target_imgs_path):
	standard = 1e+10
	batch_size = args.batch_size

	# load network model, RGB
	in_c = 1 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c
	densefuse_model = DenseFuse_net(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		densefuse_model.load_state_dict(torch.load(args.resume))
	print(densefuse_model)
	optimizer = Adam(densefuse_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		densefuse_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_source1, image_set_source2, image_set_target, batches = utils.load_dataset(source1_imgs_path, source2_imgs_path, target_imgs_path, batch_size)
		densefuse_model.train()
		count = 0
		for batch in range(batches):
			source1_paths = image_set_source1[batch * batch_size:(batch * batch_size + batch_size)]
			source2_paths = image_set_source2[batch * batch_size:(batch * batch_size + batch_size)]
			target_paths = image_set_target[batch * batch_size:(batch * batch_size + batch_size)]
			
			source1 = utils.get_train_images_auto(source1_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
			source2 = utils.get_train_images_auto(source2_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
			target = utils.get_train_images_auto(target_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
			BoEt = utils.get_train_images_auto_tb(source1_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model, type='low')
			DoEt = utils.get_train_images_auto_tb(source2_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model, type='high')

			count += 1
			optimizer.zero_grad()
			source1 = Variable(source1, requires_grad=False)
			source2 = Variable(source2, requires_grad=False)
			target = Variable(target, requires_grad=False)
			BoEt = Variable(BoEt, requires_grad=False)
			DoEt = Variable(DoEt, requires_grad=False)
            
			source1 = source1.squeeze()
			source2 = source2.squeeze()
			lambda1, lambda2 = AGC.lambda_generator(np.array(source1), np.array(source2), np.array(target.squeeze()))
            
			mask_l = torch.tensor(generate_mask(np.array(source1), 50, 'low'))
			mask_h = torch.tensor(generate_mask(np.array(source2), 50, 'high'))
            
			source1 = source1.reshape([1,1,256,256])
			source2 = source2.reshape([1,1,256,256])
    		
			if args.cuda:
				target = target.cuda()
				source2 = source2.cuda()
				source1 = source1.cuda()
				BoEt = BoEt.cuda()
				DoEt = DoEt.cuda()
				mask_l = mask_l.cuda()
				mask_h = mask_h.cuda()
                
			# get fusion image
			# encoder
			en1 = densefuse_model.encoder1(source1)
			en2 = densefuse_model.encoder2(source2)

			# feature map addition
			gen = densefuse_model.fusion(en1, en2)

			# decoder
			outputs = densefuse_model.decoder(gen)
			
            # post processing the generated image
			generated_img = torch.stack(outputs,dim = 0)
			generated_img = generated_img.squeeze()
			BoE = generated_img*mask_l
			DoE = generated_img*mask_h
            
			BoE = BoE.reshape([1,1,256,256])
			DoE = DoE.reshape([1,1,256,256])
            
			# resolution loss
			x = Variable(target.data.clone(), requires_grad=False)
			y = Variable(BoEt.data.clone(), requires_grad=False)
			z = Variable(DoEt.data.clone(), requires_grad=False)

			ssim_loss_value_med = 0.
			pixel_loss_value_med = 0.
			ssim_loss_value_low = 0.
			pixel_loss_value_low = 0.        
			ssim_loss_value_high = 0.
			pixel_loss_value_high = 0.          
            
			for output in outputs:
				pixel_loss_temp_med = mse_loss(output, x)
				ssim_loss_temp_med = ssim_loss(output, x, normalize=True)
				ssim_loss_value_med += (1-ssim_loss_temp_med)
				pixel_loss_value_med += pixel_loss_temp_med

				pixel_loss_temp_low = mse_loss(BoE, y)
				ssim_loss_temp_low = ssim_loss(BoE, y, normalize=True)
				ssim_loss_value_low += (1-ssim_loss_temp_low)
				pixel_loss_value_low += pixel_loss_temp_low

				pixel_loss_temp_high = mse_loss(DoE, z)
				ssim_loss_temp_high = ssim_loss(DoE, z, normalize=True)
				ssim_loss_value_high += (1-ssim_loss_temp_high)
				pixel_loss_value_high += pixel_loss_temp_high

			ssim_loss_value_med /= len(outputs)
			pixel_loss_value_med /= len(outputs)
			ssim_loss_value_low /= len(outputs)
			pixel_loss_value_low /= len(outputs)
			ssim_loss_value_high /= len(outputs)
			pixel_loss_value_high /= len(outputs)

			# total loss
			total_loss = (pixel_loss_value_med + args.ssim_weight[i] * ssim_loss_value_med) + lambda1*(pixel_loss_value_low + args.ssim_weight[i] * ssim_loss_value_low) + lambda2*(pixel_loss_value_high + args.ssim_weight[i] * ssim_loss_value_high)
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += (ssim_loss_value_med.item() + ssim_loss_value_low.item() + ssim_loss_value_high.item()) 
			all_pixel_loss += (pixel_loss_value_med.item() + pixel_loss_value_low.item() + pixel_loss_value_high.item()) 
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % 460000 == 0:
				# save model
				densefuse_model.eval()
				densefuse_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(densefuse_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				densefuse_model.train()
				densefuse_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
            
			if total_loss < standard:
				standard = total_loss
                
                # save model
				densefuse_model.eval()
				densefuse_model.cpu()
				save_model_filename = args.ssim_path[i] + "_best.model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(densefuse_model.state_dict(), save_model_path)
                
				densefuse_model.train()
				densefuse_model.cuda()
				print("\nbest, loss :", int(total_loss))
                

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	densefuse_model.eval()
	densefuse_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(densefuse_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
	main()
