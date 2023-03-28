import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
#from torch.utils.serialization import load_lua
from args_fusion import args
#from scipy.misc import imread, imsave, imresize
from imageio import imread, imsave
import matplotlib as mpl
import cv2
from torchvision import datasets, transforms
from blur import generate_mask



def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(source1_path, source2_path, target_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(source1_path)
    source1_imgs_path = source1_path[:num_imgs]
    source2_imgs_path = source2_path[:num_imgs]
    target_imgs_path = target_path[:num_imgs]
	# random
    data1 = np.array(source1_imgs_path)
    data2 = np.array(source2_imgs_path)
    data3 = np.array(target_imgs_path)

    s = np.arange(data1.shape[0])
    np.random.shuffle(s)
    source1_imgs_path = data1[s]
    source2_imgs_path = data2[s]          
    target_imgs_path = data3[s]
    
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source1_imgs_path = source1_imgs_path[:-mod]
        source2_imgs_path = source2_imgs_path[:-mod]
        target_imgs_path = target_imgs_path[:-mod]
        
    batches = int(len(source1_imgs_path) // BATCH_SIZE)
    return source1_imgs_path, source2_imgs_path, target_imgs_path, batches

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, pilmode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    elif mode == 'LAB':
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    if height is not None and width is not None:
        image = cv2.resize(image, (height, width), cv2.INTER_NEAREST)
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_train_images_auto_tb(paths, height=256, width=256, mode='RGB', type='none'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        mask = generate_mask(image.copy(), 10, type)
        image = image*mask

        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        elif mode =='LAB':
            
            image, a, b = cv2.split(image)
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])          
        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    if mode == 'LAB':
        return images, a, b
    else:
        return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imsave(path, data)

def save_images_lab(path, data, a1, a2, b1, b2):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    a = a1/2 + a2/2
    b = b1/2 + b2/2
 
    lab = cv2.merge([data,a.astype('uint8'),b.astype('uint8')])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(path, rgb)
