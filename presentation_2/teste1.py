import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

arquivo = "/home/ana/PycharmProjects/Visao_Computacional/presentation2/volume.mhd"

img = sitk.ReadImage(arquivo)
volume = sitk.GetArrayFromImage(img)

print("Shape:", volume.shape)
print("Mínimo:", volume.min())
print("Máximo:", volume.max())
print("Spacing:", img.GetSpacing())
print("Origin:", img.GetOrigin())
print("Direction:", img.GetDirection())

slices = [20, 40, 60, 80, 100]

plt.figure(figsize=(15, 5))

for i, s in enumerate(slices):
    plt.subplot(1, len(slices), i + 1)
    plt.imshow(volume[s], cmap="gray")
    plt.title(f"Slice {s}")
    plt.axis("off")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

slice_idx = 80

slice_img = volume[slice_idx]

# Máscara baseada em HU
mask = np.logical_and(slice_img >= -1000,
                      slice_img <= -300)

print("Valores únicos:", np.unique(mask))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(slice_img, cmap="gray")
plt.title("Slice original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Máscara inicial do pulmão")
plt.axis("off")

plt.show()


from skimage.segmentation import clear_border

mask_sem_borda = clear_border(mask)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mask, cmap="gray")
plt.title("Máscara antes")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask_sem_borda, cmap="gray")
plt.title("Máscara sem bordas")
plt.axis("off")

plt.show()

from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt

labeled = label(mask_sem_borda)
regions = regionprops(labeled)

print("Quantidade de componentes:", len(regions))

regions = sorted(regions, key=lambda r: r.area, reverse=True)

for i, r in enumerate(regions[:10]):
    print(f"Componente {i+1}: área = {r.area}")

mask_componentes = np.zeros_like(mask_sem_borda, dtype=np.uint8)

for r in regions[:2]:
    mask_componentes[labeled == r.label] = 1

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mask_sem_borda, cmap="gray")
plt.title("Após clear_border")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask_componentes, cmap="gray")
plt.title("Dois maiores componentes")
plt.axis("off")

plt.show()


from skimage.morphology import closing, disk

mask_fechada = closing(mask_componentes, disk(5))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mask_componentes, cmap="gray")
plt.title("Antes do fechamento")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask_fechada, cmap="gray")
plt.title("Após fechamento morfológico")
plt.axis("off")

plt.show()


from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

mask_preenchida = binary_fill_holes(mask_fechada)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mask_fechada, cmap="gray")
plt.title("Após fechamento")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask_preenchida, cmap="gray")
plt.title("Após preencher buracos")
plt.axis("off")

plt.show()


import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes

mask_final = np.zeros_like(volume, dtype=np.uint8)

for i in range(volume.shape[0]):
    slice_img = volume[i]

    mask = np.logical_and(slice_img >= -1000, slice_img <= -300)

    mask = clear_border(mask)

    labeled = label(mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        continue

    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    mask_componentes = np.zeros_like(mask, dtype=np.uint8)

    for r in regions[:2]:
        mask_componentes[labeled == r.label] = 1

    mask_fechada = binary_closing(mask_componentes, disk(5))

    mask_preenchida = binary_fill_holes(mask_fechada)

    mask_final[i] = mask_preenchida.astype(np.uint8)

print("Shape da máscara:", mask_final.shape)
print("Valores únicos:", np.unique(mask_final))
print("Soma total:", np.sum(mask_final))



slices = [40, 60, 80, 100]

plt.figure(figsize=(14, 7))

for idx, s in enumerate(slices):
    plt.subplot(2, len(slices), idx + 1)
    plt.imshow(volume[s], cmap="gray")
    plt.title(f"Original {s}")
    plt.axis("off")

    plt.subplot(2, len(slices), idx + 1 + len(slices))
    plt.imshow(mask_final[s], cmap="gray")
    plt.title(f"Máscara {s}")
    plt.axis("off")

plt.tight_layout()
plt.show()

volume_pulmao = volume * mask_final

print("Shape:", volume_pulmao.shape)
print("Min:", volume_pulmao.min())
print("Max:", volume_pulmao.max())



volume_pulmao = volume * mask_final

slices = [40, 60, 80, 100]

plt.figure(figsize=(14, 7))

for idx, s in enumerate(slices):

    plt.subplot(2, len(slices), idx + 1)
    plt.imshow(volume[s], cmap="gray")
    plt.title(f"Original {s}")
    plt.axis("off")

    plt.subplot(2, len(slices), idx + 1 + len(slices))
    plt.imshow(volume_pulmao[s], cmap="gray")
    plt.title(f"Pulmão segmentado {s}")
    plt.axis("off")

plt.tight_layout()
plt.show()

from skimage.measure import label, regionprops

# Componentes conectados em 3D
labeled_3d = label(mask_final, connectivity=1)
regions_3d = regionprops(labeled_3d)

print("Componentes 3D encontrados:", len(regions_3d))

regions_3d = sorted(regions_3d, key=lambda r: r.area, reverse=True)

mask_limpa_3d = np.zeros_like(mask_final, dtype=np.uint8)

# Mantém só os 2 maiores componentes 3D: pulmão esquerdo e direito
for r in regions_3d[:2]:
    mask_limpa_3d[labeled_3d == r.label] = 1

print("Soma antes:", np.sum(mask_final))
print("Soma depois:", np.sum(mask_limpa_3d))

mask_final = mask_limpa_3d