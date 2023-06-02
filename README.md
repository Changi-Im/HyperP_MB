## HyperP_MB

Code of [Multi-Task Learning Approach Using Dynamic Hyperparameter for Multi-Exposure Fusion](https://doi.org/10.3390/math11071620)

### Abstract
* High-dynamic-range (HDR) image synthesis is a technology developed to accurately reproduce the actual scene of an image on a display by extending the dynamic range of an image. Multi-exposure fusion (MEF) technology, which synthesizes multiple low-dynamic-range (LDR) images to create an HDR image, has been developed in various ways including pixel-based, patch-based, and deep learning-based methods. Recently, methods to improve the synthesis quality of images using deep-learning-based algorithms have mainly been studied in the field of MEF. Despite the various advantages of deep learning, deep-learning-based methods have a problem in that numerous multi-exposed and ground-truth images are required for training. In this study, we propose a self-supervised learning method that generates and learns reference images based on input images during the training process. In addition, we propose a method to train a deep learning model for an MEF with multiple tasks using dynamic hyperparameters on the loss functions. It enables effective network optimization across multiple tasks and high-quality image synthesis while preserving a simple network architecture. Our learning method applied to the deep learning model shows superior synthesis results compared to other existing deep-learning-based image synthesis algorithms.

### Tips
* We developed code based on [this code](https://github.com/hli1221/densefuse-pytorch)
* [Training dataset](https://knuackr-my.sharepoint.com/:u:/g/personal/imchmcgi2_knu_ac_kr/ERduPdE1J6VHjJGNRqCjf4oBFos4rZvMPTfTkdi0rQIb0Q?e=NAhvjW) 

### Citation
```
@article{math11071620,
  author = {Im, Chan-Gi and Son, Dong-Min and Kwon, Hyuk-Ju and Lee, Sung-Hak},
  title = {Multi-Task Learning Approach Using Dynamic Hyperparameter for Multi-Exposure Fusion},
  journal = {Mathematics},
  volume = {11},
  year = {2023},
  number = {7},
  article-number = {1620},
  url = {https://www.mdpi.com/2227-7390/11/7/1620},
  issn = {2227-7390},
  doi = {10.3390/math11071620}
}
```

### Contact Information
If you have any question, please email to me [imchmcgi250@gmail.com](imchmcgi250@gmail.com).
