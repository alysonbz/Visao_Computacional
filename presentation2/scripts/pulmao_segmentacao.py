import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk

img = sitk.ReadImage('/presentation2/volume.mhd')

volume = sitk.GetArrayFromImage(img)

print(volume.shape)


#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to load voxel data.
"""
from skimage.segmentation import clear_border
import re
import os
import sys

import cv2
import numpy as np

from skimage.feature import local_binary_pattern, peak_local_max

from scipy.ndimage import label

# Função para salvar os dados em mhd depois de processados
def write_ITK_metaimage(volume, name):
    img = sitk.GetImageFromArray(volume.astype(np.int16))
    sitk.WriteImage(img, name + ".mhd")

# Função para ler o arquivo mhd
def read_itk_metaimage(filename):
    img = sitk.ReadImage(filename)
    volume = sitk.GetArrayFromImage(img)
    return volume

# Função de normalização da imagem
def normalizeImage(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result


def vision3(image):
    cv2.imshow("original", normalizeImage(image))
    lbp_image = local_binary_pattern(image, 8, 1, "uniform")
    cv2.imshow("lbp", normalizeImage(lbp_image))

    image = image + np.double(lbp_image)

    return np.uint8(np.log(image / np.log(2)));


def img_fill(im_in, n):  # n = binary image threshold
    th, im_th = cv2.threshold(im_in, n, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv

    return fill_image


if __name__ == "__main__":
    #
    volume = read_itk_metaimage('/presentation2/volume.mhd')
    volume = (volume * -297.49) / np.mean(volume);
    volume = abs((volume < -40) * volume);
    print(volume.shape)

    kernel = np.ones((3, 3), np.uint8)

    visualDebug = 1

    grayFinal = np.copy(volume)
    x, y, z = grayFinal.shape
    for j in range(x):
        print("Slice Number: ", j)
        originalFrame = np.copy(grayFinal[j])
        rt = vision3(grayFinal[j])
        ty = normalizeImage(rt)

        cleared = clear_border((ty > 220) * 1)
        cv2.imshow("cleared", normalizeImage(cleared))
        ty2 = img_fill(normalizeImage(cleared), 220);
        cv2.imshow("img_fill", normalizeImage(ty2))

        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(normalizeImage(ty2), 4, cv2.CV_32S)

        grayFinal[j] = ty2;
        img2 = np.zeros((grayFinal[j].shape))
        if len(stats[1:, -1]) > 2:
            ind = np.argpartition(stats[1:, -1], -2)[-2:]
            ind = ind + 1

            for i in ind:
                img2[labels == i] = 255

            grayFinal[j] = img2;
            k = cv2.waitKey(0)

        elif len(stats[1:, -1]) > 1:
            ind = np.argpartition(stats[1:, -1], -2)[-2:]
            ind = ind + 1
            for i in ind:
                img2[labels == i] = 255
            grayFinal[j] = img2;

        if visualDebug == 1:
            vis = np.concatenate((normalizeImage(ty), normalizeImage(cleared)), axis=1)
            cv2.imshow('1y32', vis)
            k = cv2.waitKey(0)

    labeled_array, num_features = label(grayFinal)

    resultSegmentation = np.zeros((grayFinal.shape))

    g = 0
    for i in range(num_features):
        if i != 0:
            volCopy = np.zeros((grayFinal.shape))

            volCopy[labeled_array == i] = 255;
            if np.sum(volCopy) > 100000000:
                g = np.sum(volCopy)
                print(np.sum(volCopy), i)
                resultSegmentation = resultSegmentation + np.copy(volCopy);

    write_ITK_metaimage(resultSegmentation, "rt6")