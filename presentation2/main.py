import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, label, binary_dilation, binary_erosion, binary_closing
from skimage.morphology import remove_small_objects
from skimage.filters import frangi
from skimage.transform import resize
import os
import pandas as pd

# ── Configurações ─────────────────────────────────────────────────────────────
DATA_DIR = "train"  # pasta com os volumes
OUTPUT_CSV = "resultados.csv"
FRANGI_FATOR = 0.5          # 256x256
FRANGI_THRESHOLD = 0.0003
FRANGI_SIGMAS = range(1, 5)

# ── Funções ───────────────────────────────────────────────────────────────────
def criar_mascara_pulmao_slice(fatia):
    mascara_corpo = binary_fill_holes(fatia > 0.10)
    mascara_corpo = remove_small_objects(mascara_corpo, max_size=5000)
    labeled_corpo, n = label(mascara_corpo)
    if n == 0:
        return np.zeros_like(fatia, dtype=bool)
    tamanhos = [(labeled_corpo == i).sum() for i in range(1, n+1)]
    mascara_corpo = labeled_corpo == (np.argmax(tamanhos) + 1)
    mascara_ar = (fatia < 0.20) & mascara_corpo
    mascara_ar = remove_small_objects(mascara_ar, max_size=3000)
    if mascara_ar.sum() == 0:
        return np.zeros_like(fatia, dtype=bool)
    labeled_ar, n_ar = label(mascara_ar)
    tamanhos_ar = [(labeled_ar == i).sum() for i in range(1, n_ar+1)]
    idx_ord = np.argsort(tamanhos_ar)[::-1]
    mascara_final = np.zeros_like(fatia, dtype=bool)
    for idx in idx_ord[:2]:
        comp = labeled_ar == (idx + 1)
        comp = binary_fill_holes(comp)
        if comp.sum() > 1000:
            mascara_final |= comp
    if mascara_final.sum() < 1000:
        return np.zeros_like(fatia, dtype=bool)
    return mascara_final

def criar_mascara_pulmao(img_norm):
    H, W, Z = img_norm.shape
    mascara = np.zeros((H, W, Z), dtype=bool)
    for z in range(Z):
        mascara[:, :, z] = criar_mascara_pulmao_slice(img_norm[:, :, z].T).T
    return mascara

def dilatar_mascara(mascara, iterations=5):
    struct = np.ones((3, 3, 3), dtype=bool)
    return binary_dilation(mascara, structure=struct, iterations=iterations)

def segmentar_volume(ct_volume, mask_volume):
    # 1. Pré-processamento
    img_clip = np.clip(ct_volume, -1000, 600)
    img_norm = (img_clip - (-1000)) / (600 - (-1000))

    # 2. Máscara do pulmão + dilatação
    mascara_pulmao = criar_mascara_pulmao(img_norm)
    mascara_roi = dilatar_mascara(mascara_pulmao, iterations=5)

    # 3. Aplica ROI
    img_roi = img_norm.copy()
    img_roi[~mascara_roi] = 0

    # 4. Frangi 3D em resolução reduzida
    H, W, Z = img_roi.shape
    Hr, Wr = int(H * FRANGI_FATOR), int(W * FRANGI_FATOR)
    img_pequena = resize(img_roi, (Hr, Wr, Z), anti_aliasing=True)
    frangi_3d = frangi(
        img_pequena,
        sigmas=FRANGI_SIGMAS,
        alpha=0.5,
        beta=0.5,
        gamma=2,
        black_ridges=False
    )
    frangi_volume = resize(frangi_3d, (H, W, Z), anti_aliasing=True)

    # 5. Threshold + pós-processamento
    seg = frangi_volume > FRANGI_THRESHOLD
    seg = remove_small_objects(seg, max_size=10)
    seg = binary_closing(seg, iterations=1)

    # 6. Métricas
    gt = mask_volume > 0
    tp = (seg & gt).sum()
    fp = (seg & ~gt).sum()
    fn = (~seg & gt).sum()
    dice      = 2*tp / (2*tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    return seg, dice, precision, recall

# ── Loop principal ────────────────────────────────────────────────────────────
casos = sorted(os.listdir(DATA_DIR))
resultados = []

for i, caso in enumerate(casos):
    ct_path   = os.path.join(DATA_DIR, caso, "image", f"{caso}.nii.gz")
    mask_path = os.path.join(DATA_DIR, caso, "label", f"{caso}.nii.gz")

    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        print(f"[{i+1}/{len(casos)}] {caso} — arquivos não encontrados, pulando")
        continue

    print(f"[{i+1}/{len(casos)}] Processando {caso}...")
    try:
        ct_volume   = nib.load(ct_path).get_fdata()
        mask_volume = nib.load(mask_path).get_fdata()

        seg, dice, precision, recall = segmentar_volume(ct_volume, mask_volume)

        resultados.append({
            "caso": caso,
            "dice": round(dice, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        })
        print(f"  Dice={dice:.3f}  Precision={precision:.3f}  Recall={recall:.3f}")

    except Exception as e:
        print(f"  ERRO: {e}")
        resultados.append({"caso": caso, "dice": None, "precision": None, "recall": None})

# ── Salva resultados ──────────────────────────────────────────────────────────
df = pd.DataFrame(resultados)
df.to_csv(OUTPUT_CSV, index=False)

print("\n── Resultado Final ──────────────────────────────────────────────")
print(f"Casos processados: {df['dice'].notna().sum()}/{len(df)}")
print(f"Dice médio:        {df['dice'].mean():.3f} ± {df['dice'].std():.3f}")
print(f"Precision média:   {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
print(f"Recall médio:      {df['recall'].mean():.3f} ± {df['recall'].std():.3f}")
print(f"\nResultados salvos em: {OUTPUT_CSV}")