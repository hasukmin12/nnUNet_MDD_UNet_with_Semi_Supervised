# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    Draw the Class Activation Map
         :param model: Pytorch model with weights loaded
         :param img_path: test image path
         :param save_path: CAM result save path
         :param transform: input image preprocessing method
         :param visual_heatmap: Whether to visualize the original heatmap (call matplotlib)
    :return:
    '''
    # Image loading & preprocessing
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # Get the feature/score of the model output
    model.eval()
    features = model.features(img)
    output = model.classifier(features)

    # In order to be able to read the auxiliary function defined by the intermediate gradient
    def extract(g):
        global features_grad
        features_grad = g

        # Predict the output score corresponding to the category with the highest score

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # calculate the gradient

    grads = features_grad  # Get gradient


pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

# Here the batch size defaults to 1, so the 0th dimension (batch size dimension) is removed
pooled_grads = pooled_grads[0]
features = features[0]
# 512 is the number of channels in the last layer of feature
for i in range(512):
    features[i, ...] *= pooled_grads[i, ...]

    # The following parts are implemented with Keras version
heatmap = features.detach().numpy()
heatmap = np.mean(heatmap, axis=0)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Visualize the original heat map
if visual_heatmap:
    plt.matshow(heatmap)
    plt.show()

    img = cv2.imread(img_path)  # Load the original image with cv2
    heatmap = cv2.resize(heatmap, (
    img.shape[1], img.shape[0]))  # Adjust the size of the heat map to be the same as the original image
    heatmap = np.uint8(255 * heatmap)  # Convert the heat map to RGB format
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply the heat map to the original image
    superimposed_img = heatmap * 0.4 + img  # here 0.4 is the heat map intensity factor
    cv2.imwrite(save_path, superimposed_img)  # save the image to the hard disk