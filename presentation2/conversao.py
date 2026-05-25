import numpy as np
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import SimpleITK as sitk
import os


# Pasta contendo os arquivos DICOM
pasta_dicom = '/home/ana/PycharmProjects/Visao_Computacional/presentation2/lidc_idri/LIDC-IDRI-0001/30178/03192'

# Nome do arquivo de saída
saida_mhd = "volume.mhd"

# Leitor da série DICOM
reader = sitk.ImageSeriesReader()

# Obtém os nomes dos arquivos DICOM
arquivos_dicom = reader.GetGDCMSeriesFileNames(pasta_dicom)

# Define os arquivos
reader.SetFileNames(arquivos_dicom)

# Lê o volume 3D
imagem = reader.Execute()

# Salva como .mhd
sitk.WriteImage(imagem, saida_mhd)

print("Conversão concluída!")