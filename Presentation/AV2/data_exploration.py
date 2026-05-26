import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

base_path = "AeroPath/1"
ct_file = os.path.join(base_path, "1_CT_HR.nii.gz")
airway_mask_file = os.path.join(base_path, "1_CT_HR_label_airways.nii.gz")


def carregar_e_visualizar():
    print(f"Carregando imagens do diretório: {base_path}...")

    # Leitura dos volumes 3D usando nibabel
    ct_img = nib.load(ct_file)
    mask_img = nib.load(airway_mask_file)

    # Extraindo os dados em formato de array do NumPy
    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    print(f"Dimensões do volume CT: {ct_data.shape}")

    # 3. Seleção de uma fatia (Slice) para visualização 2D
    # Imagens 3D geralmente possuem o formato (x, y, z).
    # Vamos pegar uma fatia transversal (axial) bem no meio do eixo Z.
    z_slice = ct_data.shape[2] // 2

    # Extraindo as matrizes 2D
    ct_slice = ct_data[:, :, z_slice]
    mask_slice = mask_data[:, :, z_slice]

    # Rotação de 90 graus (geralmente necessária ao ler com nibabel para orientação correta)
    ct_slice = np.rot90(ct_slice)
    mask_slice = np.rot90(mask_slice)

    # 4. Visualização com Matplotlib
    plt.figure(figsize=(15, 5))

    # Plot 1: CT Original
    plt.subplot(1, 3, 1)
    # vmin e vmax ajudam a focar na janela de densidade do pulmão/ar (Unidades Hounsfield)
    plt.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=400)
    plt.title(f'Tomografia (Fatia Axial Z={z_slice})')
    plt.axis('off')

    # Plot 2: Máscara das Vias Aéreas (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_slice, cmap='gray')
    plt.title('Máscara: Vias Aéreas')
    plt.axis('off')

    # Plot 3: Sobreposição (Overlay)
    plt.subplot(1, 3, 3)
    plt.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=400)

    # Mascarando os zeros para que apenas a anotação das vias aéreas fique colorida por cima do CT
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(masked_overlay, cmap='autumn', alpha=0.6)
    plt.title('Sobreposição (CT + Máscara)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    carregar_e_visualizar()