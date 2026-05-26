import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation, ball


def calcular_metricas(mascara_predita, mascara_ideal, paciente_id):
    pred = mascara_predita > 0
    gt = mascara_ideal > 0

    intersecao = np.sum(pred & gt)
    uniao = np.sum(pred | gt)
    soma_areas = np.sum(pred) + np.sum(gt)

    if soma_areas == 0:
        return 0.0, 0.0

    dice = (2.0 * intersecao) / soma_areas
    iou = intersecao / uniao if uniao > 0 else 0.0

    print(f"Paciente {paciente_id:02d} | Dice: {dice:.4f} | IoU: {iou:.4f} | Interseção: {intersecao}")
    return dice, iou


def processar_paciente(paciente_id):
    print(f"\n[{paciente_id}/27] Processando Paciente {paciente_id}...")

    # Caminhos dinâmicos
    base_path = f"AeroPath/{paciente_id}"
    ct_file = os.path.join(base_path, f"{paciente_id}_CT_HR.nii.gz")
    airway_mask_file = os.path.join(base_path, f"{paciente_id}_CT_HR_label_airways.nii.gz")

    if not os.path.exists(ct_file) or not os.path.exists(airway_mask_file):
        print(f"  -> Arquivos não encontrados para o Paciente {paciente_id}. Pulando.")
        return None, None

    ct_data = nib.load(ct_file).get_fdata()
    ground_truth = nib.load(airway_mask_file).get_fdata()

    # Inicialização Automática
    z_seed = int(ct_data.shape[2] * 0.85)
    fatia_superior = ct_data[:, :, z_seed]

    mascara_ar = fatia_superior < -800
    propriedades = regionprops(label(mascara_ar))

    centro_x, centro_y = fatia_superior.shape[0] // 2, fatia_superior.shape[1] // 2
    melhor_candidato = None
    menor_distancia = float('inf')

    for prop in propriedades:
        if 50 < prop.area < 3000:
            cx, cy = prop.centroid
            dist_centro = ((cx - centro_x) ** 2 + (cy - centro_y) ** 2) ** 0.5
            if dist_centro < menor_distancia:
                menor_distancia = dist_centro
                melhor_candidato = prop

    if melhor_candidato is None:
        print(f"  -> Falha na inicialização da traqueia. Pulando.")
        return None, None

    cx, cy = melhor_candidato.centroid
    seed_point = (int(cx), int(cy), z_seed)

    # Execução do Crescimento de Regiões
    rg_mask = flood(ct_data, seed_point, tolerance=70)

    # Pos-processamento
    raio_erosao = 1
    max_raio = 5
    volume_maximo_esperado = 600000
    arvore_isolada = None

    while raio_erosao <= max_raio:
        elemento_estruturante = ball(raio_erosao)
        mask_erodida = erosion(rg_mask, footprint=elemento_estruturante)
        ilhas_rotuladas = label(mask_erodida)

        rotulo_da_traqueia = ilhas_rotuladas[seed_point]

        # Se apagou a traqueia, volta um passo e para
        if rotulo_da_traqueia == 0:
            raio_erosao -= 1
            break

        arvore_isolada_temp = (ilhas_rotuladas == rotulo_da_traqueia)
        volume_atual = np.sum(arvore_isolada_temp)

        if volume_atual < volume_maximo_esperado:
            arvore_isolada = arvore_isolada_temp
            break
        else:
            arvore_isolada = arvore_isolada_temp
            raio_erosao += 1

    # Rede de Segurança caso o raio 1 já apague a traqueia
    if arvore_isolada is None:
        ilhas_originais = label(rg_mask)
        arvore_isolada = (ilhas_originais == ilhas_originais[seed_point])
        raio_erosao = 0

    if raio_erosao > 0:
        elemento_estruturante_final = ball(raio_erosao)
        mask_final = dilation(arvore_isolada, footprint=elemento_estruturante_final)
    else:
        mask_final = arvore_isolada

    # Cálculo de Métricas
    dice, iou = calcular_metricas(mask_final, ground_truth, paciente_id)

    # Visualização Dinâmica (Z PADRONIZADO EM 383)
    z_vis = min(383, ct_data.shape[2] - 1)

    ct_slice_sup = np.rot90(ct_data[:, :, z_seed])
    rg_slice_sup = np.rot90(rg_mask[:, :, z_seed])

    ct_slice_inf = np.rot90(ct_data[:, :, z_vis])
    gt_slice_inf = np.rot90(ground_truth[:, :, z_vis])
    rg_slice_vazada = np.rot90(rg_mask[:, :, z_vis])
    mask_final_slice = np.rot90(mask_final[:, :, z_vis])

    coords = np.argwhere((gt_slice_inf > 0) | (mask_final_slice > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        borda = 30
        y_min, y_max = max(0, y_min - borda), min(ct_slice_inf.shape[0], y_max + borda)
        x_min, x_max = max(0, x_min - borda), min(ct_slice_inf.shape[1], x_max + borda)

        gt_crop = gt_slice_inf[y_min:y_max, x_min:x_max]
        pred_crop = mask_final_slice[y_min:y_max, x_min:x_max]
    else:
        gt_crop, pred_crop = gt_slice_inf, mask_final_slice

    plt.figure(figsize=(25, 5))

    # Plot 1
    plt.subplot(1, 5, 1)
    plt.imshow(ct_slice_sup, cmap='gray', vmin=-1000, vmax=400)
    plt.imshow(np.ma.masked_where(rg_slice_sup == 0, rg_slice_sup), cmap='spring', alpha=0.6)
    plt.title(f'1. Traqueia (Z={z_seed})')
    plt.axis('off')

    # Plot 2
    plt.subplot(1, 5, 2)
    plt.imshow(ct_slice_inf, cmap='gray', vmin=-1000, vmax=400)
    plt.imshow(np.ma.masked_where(rg_slice_vazada == 0, rg_slice_vazada), cmap='spring', alpha=0.5)
    plt.title(f'2. Vazamento (Z={z_vis})')
    plt.axis('off')

    # Plot 3
    plt.subplot(1, 5, 3)
    plt.imshow(ct_slice_inf, cmap='gray', vmin=-1000, vmax=400)
    plt.imshow(np.ma.masked_where(mask_final_slice == 0, mask_final_slice), cmap='spring', alpha=0.6)
    plt.title(f'3. Corrigido com ball({raio_erosao})')
    plt.axis('off')

    # Plot 4
    plt.subplot(1, 5, 4)
    plt.imshow(gt_slice_inf, cmap='gray')
    plt.title('4. Máscara Ideal')
    plt.axis('off')

    # Plot 5
    plt.subplot(1, 5, 5)
    plt.imshow(gt_crop, cmap='gray')
    plt.imshow(np.ma.masked_where(pred_crop == 0, pred_crop), cmap='spring', alpha=0.6)
    plt.title(f'5. Zoom: Predição vs GT (Z={z_vis})')
    plt.axis('off')

    plt.tight_layout()

    # Salva a imagem ao invés de mostrar na tela
    nome_arquivo = f"resultados_visuais/paciente_{paciente_id:02d}.png"
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=150)
    plt.close()  # Limpa a memória para o próximo paciente

    return dice, iou


def executar_pipeline_em_lote():
    # Cria a pasta para salvar as imagens, se não existir
    os.makedirs("resultados_visuais", exist_ok=True)

    todos_dice = []
    todos_iou = []

    print("==================================================")
    print(" INICIANDO PROCESSAMENTO EM LOTE (27 PACIENTES) ")
    print("==================================================")

    for i in range(1, 28):
        try:
            dice, iou = processar_paciente(i)
            if dice is not None and iou is not None:
                todos_dice.append(dice)
                todos_iou.append(iou)
        except Exception as e:
            print(f"  -> Erro crítico no Paciente {i}: {e}")

    if len(todos_dice) > 0:
        media_dice = np.mean(todos_dice)
        media_iou = np.mean(todos_iou)
        desvio_dice = np.std(todos_dice)

        print("\n==================================================")
        print(" RESUMO FINAL DA AVALIAÇÃO (TODOS OS PACIENTES) ")
        print("==================================================")
        print(f"Total de Pacientes Processados: {len(todos_dice)}/27")
        print(f"MÉDIA DICE: {media_dice:.4f} ± {desvio_dice:.4f}")
        print(f"MÉDIA IoU:  {media_iou:.4f}")
        print("Todas as imagens foram salvas na pasta 'resultados_visuais'.")
        print("==================================================")


if __name__ == "__main__":
    executar_pipeline_em_lote()