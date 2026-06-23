
import numpy as np
import pandas as pd
import pydicom
import joblib
from pathlib import Path
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops


# CONFIGURAÇÃO

PATCH_SIZE = 32          # voxels — cobre nódulos de até ~22mm (ver relatório)
HU_MIN_PULMAO = -1000
HU_MAX_PULMAO = 400

# diretório onde os modelos treinados foram salvos (joblib .pkl)
MODEL_DIR = Path(__file__).parent / "models"


# ETAPA 1 — Carregamento do volume DICOM

def _carregar_volume(exame_dir: Path):

    dcm_files = sorted(Path(exame_dir).rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"Nenhum arquivo .dcm encontrado em {exame_dir}")

    slices = []
    for f in dcm_files:
        try:
            slices.append(pydicom.dcmread(str(f)))
        except Exception:
            continue
    if not slices:
        raise ValueError(f"Não foi possível ler nenhum DICOM válido em {exame_dir}")

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    ds0       = slices[0]
    slope     = float(getattr(ds0, "RescaleSlope", 1))
    intercept = float(getattr(ds0, "RescaleIntercept", 0))
    volume_hu = volume * slope + intercept

    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    spacing_xy  = float(slices[0].PixelSpacing[0])

    meta = {"z_positions": z_positions, "spacing_xy": spacing_xy}
    return volume_hu, meta


# ETAPA 2 — Localização e recorte do patch


def _coordenada_para_voxel(x, y, z, meta):

    vx = int(round(x))
    vy = int(round(y))
    diffs = [abs(zp - z) for zp in meta["z_positions"]]
    vz = int(np.argmin(diffs))
    return vx, vy, vz


def _recortar_patch(volume_hu, vx, vy, vz, size=PATCH_SIZE):
    """Recorta um cubo size x size x size centrado no voxel do nódulo."""
    h = size // 2
    D, H, W = volume_hu.shape
    z0, z1 = max(0, vz - h), min(D, vz + h)
    y0, y1 = max(0, vy - h), min(H, vy + h)
    x0, x1 = max(0, vx - h), min(W, vx + h)

    patch = volume_hu[z0:z1, y0:y1, x0:x1]
    pad = (
        (h - (vz - z0), h - (z1 - vz)),
        (h - (vy - y0), h - (y1 - vy)),
        (h - (vx - x0), h - (x1 - vx)),
    )
    patch = np.pad(patch, pad, mode="constant", constant_values=-1000)
    return patch


# ETAPA 3 — Segmentação automática (threshold HU + connected components)

def _segmentar_nodulo(patch, hu_min=-300, hu_max=400, max_fracao=0.4):

    mascara_densidade = (patch > hu_min) & (patch < hu_max)
    labels, _ = ndimage.label(mascara_densidade)
    centro = tuple(s // 2 for s in patch.shape)
    label_central = labels[centro]

    if label_central == 0:
        return mascara_densidade

    mascara = labels == label_central
    fracao = mascara.sum() / mascara.size

    tentativas = 0
    while fracao > max_fracao and tentativas < 3:
        mascara_erodida = ndimage.binary_erosion(mascara, iterations=1)
        if mascara_erodida[centro] == 0:
            break
        labels2, _ = ndimage.label(mascara_erodida)
        label_central2 = labels2[centro]
        mascara = labels2 == label_central2
        fracao = mascara.sum() / mascara.size
        tentativas += 1

    return mascara


# ETAPA 4 — Extração de features (shape + GLCM + intensidade)

def _features_shape(mascara):
    volume_voxels = mascara.sum()
    if volume_voxels == 0:
        return {
            "shape_volume": 0, "shape_superficie": 0,
            "shape_esfericidade": 0, "shape_compacidade": 0,
            "shape_extensao_z": 0, "shape_extensao_y": 0, "shape_extensao_x": 0,
        }

    erodida = ndimage.binary_erosion(mascara)
    borda = mascara & ~erodida
    superficie_voxels = borda.sum()

    if superficie_voxels > 0:
        esfericidade = (np.pi ** (1/3)) * ((6 * volume_voxels) ** (2/3)) / superficie_voxels
    else:
        esfericidade = 0

    coords = np.argwhere(mascara)
    extensoes = coords.max(axis=0) - coords.min(axis=0) + 1
    raio_equivalente = max(extensoes) / 2
    volume_esfera_envolvente = (4/3) * np.pi * (raio_equivalente ** 3)
    compacidade = volume_voxels / volume_esfera_envolvente if volume_esfera_envolvente > 0 else 0

    return {
        "shape_volume":        int(volume_voxels),
        "shape_superficie":    int(superficie_voxels),
        "shape_esfericidade":  round(float(esfericidade), 4),
        "shape_compacidade":   round(float(compacidade), 4),
        "shape_extensao_z":    int(extensoes[0]),
        "shape_extensao_y":    int(extensoes[1]),
        "shape_extensao_x":    int(extensoes[2]),
    }


def _features_textura_glcm(patch, mascara):
    z_central  = patch.shape[0] // 2
    fatia      = patch[z_central]
    fatia_mask = mascara[z_central]

    fatia_clip = np.clip(fatia, -1000, 400)
    fatia_norm = ((fatia_clip + 1000) / 1400 * 255).astype(np.uint8)

    fatia_norm_masked = fatia_norm.copy()
    fatia_norm_masked[~fatia_mask] = 0

    if fatia_mask.sum() < 4:
        return {f"glcm_{p}": 0.0 for p in
                ["mean", "variance", "energy", "entropy", "contrast",
                 "correlation", "homogeneity"]}

    glcm = graycomatrix(
        fatia_norm_masked,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast    = graycoprops(glcm, "contrast").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    energy      = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()

    glcm_mean_matrix = glcm[:, :, 0, :].mean(axis=2)
    i_idx, j_idx = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    mean_val = (i_idx * glcm_mean_matrix).sum()
    var_val  = (((i_idx - mean_val) ** 2) * glcm_mean_matrix).sum()
    entropy_val = -(glcm_mean_matrix[glcm_mean_matrix > 0] *
                     np.log(glcm_mean_matrix[glcm_mean_matrix > 0])).sum()

    return {
        "glcm_mean":        round(float(mean_val), 4),
        "glcm_variance":    round(float(var_val), 4),
        "glcm_energy":      round(float(energy), 4),
        "glcm_entropy":     round(float(entropy_val), 4),
        "glcm_contrast":    round(float(contrast), 4),
        "glcm_correlation": round(float(correlation), 4),
        "glcm_homogeneity": round(float(homogeneity), 4),
    }


def _features_intensidade(patch, mascara):
    valores = patch[mascara.astype(bool)]
    if len(valores) == 0:
        return {"hu_media": 0, "hu_desvio": 0, "hu_min": 0, "hu_max": 0}
    return {
        "hu_media":  round(float(valores.mean()), 2),
        "hu_desvio": round(float(valores.std()), 2),
        "hu_min":    round(float(valores.min()), 2),
        "hu_max":    round(float(valores.max()), 2),
    }


def _extrair_features(patch, mascara):
    feats = {}
    feats.update(_features_shape(mascara))
    feats.update(_features_textura_glcm(patch, mascara))
    feats.update(_features_intensidade(patch, mascara))
    return feats


# ETAPA 5 — Carregamento do modelo treinado

_modelo_cache = {}

def _carregar_modelo(nome_modelo="rf"):

    if nome_modelo in _modelo_cache:
        return _modelo_cache[nome_modelo]

    arquivo_modelo = {
        "rf":  MODEL_DIR / "modelo_rf.pkl",
        "svm": MODEL_DIR / "modelo_svm.pkl",
    }.get(nome_modelo)

    if arquivo_modelo is None or not arquivo_modelo.exists():
        raise FileNotFoundError(
            f"Modelo '{nome_modelo}' não encontrado em {MODEL_DIR}. "
            f"Execute o treinamento antes de usar classificar_nodulos()."
        )

    modelo       = joblib.load(arquivo_modelo)
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

    _modelo_cache[nome_modelo] = (modelo, scaler, feature_cols)
    return modelo, scaler, feature_cols


# ----------- INTERFACE ---------------

def classificar_nodulos(exame_tc: str, lista_coordenadas: list, modelo: str = "rf") -> dict:

    exame_path = Path(exame_tc)

    volume_hu, meta = _carregar_volume(exame_path)


    clf, scaler, feature_cols = _carregar_modelo(modelo)

    resultados_nodulos = []

    for nodulo in lista_coordenadas:
        nodulo_id = nodulo["id"]
        x, y, z   = nodulo["x"], nodulo["y"], nodulo["z"]

        # ETAPA 2 — localiza e recorta o patch
        vx, vy, vz = _coordenada_para_voxel(x, y, z, meta)
        patch      = _recortar_patch(volume_hu, vx, vy, vz, size=PATCH_SIZE)

        # ETAPA 3 — segmentação automática 
        mascara = _segmentar_nodulo(patch)

        # ETAPA 4 — extração de features
        feats = _extrair_features(patch, mascara)

        # monta o vetor de features na MESMA ORDEM usada no treino
        X = np.array([[feats[col] for col in feature_cols]])

        if modelo == "svm":
            X_input = scaler.transform(X)
        else:
            X_input = X

        proba = clf.predict_proba(X_input)[0]
        classe_predita = clf.predict(X_input)[0]

        classe_str    = "maligno" if classe_predita == 1 else "benigno"
        probabilidade = float(proba[classe_predita])

        resultados_nodulos.append({
            "id": nodulo_id,
            "coordenada": {"x": x, "y": y, "z": z},
            "classe_predita": classe_str,
            "probabilidade": round(probabilidade, 4),
        })

    return {
        "exame": exame_path.name,
        "nodulos": resultados_nodulos,
    }

#executar essa interface 

if __name__ == "__main__":
    exemplo = classificar_nodulos(
        exame_tc=r"C:\lidc_data\dicom\LIDC-IDRI-0001",
        lista_coordenadas=[
            {"id": "nodulo_1", "x": 320, "y": 280, "z": -150.0},
        ],
        modelo="rf",
    )
    import json
    print(json.dumps(exemplo, indent=2, ensure_ascii=False))