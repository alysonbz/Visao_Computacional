# ============================================================================
# CENÁRIO 4: CLASSIFICAÇÃO DE NÓDULOS PULMONARES COM 4 MODELOS
# ============================================================================

# %% [markdown]
# ## 1. IMPORTS E CONFIGURAÇÃO

# %%
import numpy as np
np.int = int

import configparser
configparser.SafeConfigParser = configparser.ConfigParser

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pylidc as pl
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# %%
print("✓ Imports realizados com sucesso")

# %% [markdown]
# ## 2. CARREGAMENTO DE SCANS E FUNÇÕES AUXILIARES

# %%
scans = pl.query(pl.Scan).all()
print(f"Total de exames: {len(scans)}")

BASE_DICOM = "/home/ana/PycharmProjects/Visao_Computacional/presentation3/lidc_idri"

# %% [markdown]
# ## 3. DEFINIÇÃO DAS FUNÇÕES

# %%
def obter_rotulo_cenario4(cluster):
    """
    Obtém o rótulo do Cenário 4 para um nódulo (cluster).
    C4: Todos os radiologistas que concordam com a malignidade, 
    excluindo os indeterminados (classe 3).
    Retorna 'benigno', 'maligno' ou None.
    """
    malignidades = [ann.malignancy for ann in cluster]
    
    # Remove os indeterminados
    malignidades = [m for m in malignidades if m != 3]
    
    # Ninguém sobrou
    if len(malignidades) == 0:
        return None
    
    classes = []
    for m in malignidades:
        if m in [1, 2]:
            classes.append("benigno")
        elif m in [4, 5]:
            classes.append("maligno")
    
    # Todos concordam?
    if len(set(classes)) == 1:
        return classes[0]
    
    return None

# %%
def construir_dataset_scan(scan):
    """
    Constrói um DataFrame com os nódulos do Cenário 4 de um scan.
    """
    clusters = scan.cluster_annotations()
    dados = []
    
    for cluster in clusters:
        rotulo = obter_rotulo_cenario4(cluster)
        
        if rotulo is None:
            continue
        
        centroide = np.mean(
            [ann.centroid for ann in cluster],
            axis=0
        )
        
        x, y, z = centroide.astype(int)
        
        dados.append({
            "paciente": scan.patient_id,
            "x": x,
            "y": y,
            "z": z,
            "rotulo": rotulo
        })
    
    return pd.DataFrame(dados)

# %%
def carregar_volume_paciente(patient_id):
    """
    Carrega o volume DICOM de um paciente.
    Seleciona a série com maior número de arquivos DICOM.
    """
    pasta_paciente = os.path.join(BASE_DICOM, patient_id)
    
    if not os.path.exists(pasta_paciente):
        return None
    
    melhor_serie = None
    maior_numero = 0
    
    for raiz, dirs, files in os.walk(pasta_paciente):
        dcm = [f for f in files if f.endswith(".dcm")]
        
        if len(dcm) > maior_numero:
            maior_numero = len(dcm)
            melhor_serie = raiz
    
    if melhor_serie is None:
        return None
    
    try:
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(melhor_serie)
        reader.SetFileNames(series)
        img = reader.Execute()
        volume = sitk.GetArrayFromImage(img)
        return volume
    except:
        return None

# %%
def extrair_features(volume, x, y, z, tamanho=16):
    """
    Extrai 9 features estatísticas de um patch 3D ao redor do nódulo.
    Features: mean, std, min, max, median, p25, p75, energia, entropia
    """
    z1 = max(0, z - tamanho)
    z2 = min(volume.shape[0], z + tamanho)
    
    y1 = max(0, y - tamanho)
    y2 = min(volume.shape[1], y + tamanho)
    
    x1 = max(0, x - tamanho)
    x2 = min(volume.shape[2], x + tamanho)
    
    patch = volume[z1:z2, y1:y2, x1:x2]
    
    if patch.size == 0:
        return None
    
    f = {}
    f["mean"] = float(patch.mean())
    f["std"] = float(patch.std())
    f["min"] = int(patch.min())
    f["max"] = int(patch.max())
    f["median"] = float(np.median(patch))
    f["p25"] = float(np.percentile(patch, 25))
    f["p75"] = float(np.percentile(patch, 75))
    
    f["energia"] = np.mean(patch.astype(np.float32) ** 2)
    
    f["entropia"] = -np.sum(
        (np.histogram(patch, bins=32)[0] / patch.size + 1e-10)
        * np.log2(np.histogram(patch, bins=32)[0] / patch.size + 1e-10)
    )
    
    return f

# %% [markdown]
# ## 4. CONSTRUÇÃO DO DATASET COMPLETO DO CENÁRIO 4

# %%
print("Etapa 1: Coletando coordenadas de todos os nódulos do Cenário 4...")

todos_dados = []
erros = 0

for i, scan in enumerate(scans):
    try:
        df_aux = construir_dataset_scan(scan)
        if len(df_aux) > 0:
            todos_dados.append(df_aux)
        
        if (i + 1) % 50 == 0:
            print(f"  → Processados {i + 1}/{len(scans)} scans")
    except Exception as e:
        erros += 1

df_nodulos_c4 = pd.concat(todos_dados, ignore_index=True)

print(f"✓ Total de nódulos C4 coletados: {len(df_nodulos_c4)}")
print(f"  Benignos: {len(df_nodulos_c4[df_nodulos_c4['rotulo'] == 'benigno'])}")
print(f"  Malignos: {len(df_nodulos_c4[df_nodulos_c4['rotulo'] == 'maligno'])}")
print(f"  Erros: {erros}")

# %%
print("\nEtapa 2: Extraindo features de todos os nódulos...")

dados_features = []
volumes_cache = {}
processados = 0
falhados = 0

for idx, linha in df_nodulos_c4.iterrows():
    try:
        patient_id = linha["paciente"]
        
        # Cache de volumes para evitar recarregar
        if patient_id not in volumes_cache:
            volume = carregar_volume_paciente(patient_id)
            if volume is None:
                falhados += 1
                continue
            volumes_cache[patient_id] = volume
        else:
            volume = volumes_cache[patient_id]
        
        x = int(linha["x"])
        y = int(linha["y"])
        z = int(linha["z"])
        
        # Verifica se a coordenada está dentro do volume
        if z >= volume.shape[0] or y >= volume.shape[1] or x >= volume.shape[2]:
            falhados += 1
            continue
        
        f = extrair_features(volume, x, y, z)
        
        if f is None:
            falhados += 1
            continue
        
        f["paciente"] = patient_id
        f["x"] = x
        f["y"] = y
        f["z"] = z
        f["rotulo"] = linha["rotulo"]
        
        dados_features.append(f)
        processados += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  → Processados {idx + 1}/{len(df_nodulos_c4)} nódulos")
    
    except Exception as e:
        falhados += 1
        continue

df_c4 = pd.DataFrame(dados_features)

print(f"\n✓ Features extraídas com sucesso!")
print(f"  Dataset final: {df_c4.shape[0]} nódulos × {df_c4.shape[1]} features")
print(f"  Processados: {processados}")
print(f"  Falhados: {falhados}")
print(f"\nDistribuição das classes:")
print(df_c4["rotulo"].value_counts())

# %%
print("\nPrimeiras amostras do dataset:")
print(df_c4.head())

# %% [markdown]
# ## 5. PREPARAÇÃO DOS DADOS

# %%
X = df_c4.drop(columns=["paciente", "x", "y", "z", "rotulo"])
y = df_c4["rotulo"]

print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"\nFeatures: {list(X.columns)}")

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Classes codificadas: {np.unique(y_encoded)}")
print(f"Mapeamento: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print(f"Dados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")
print(f"\nDistribuição do treino:")
print(pd.Series(y_train).value_counts())
print(f"\nDistribuição do teste:")
print(pd.Series(y_test).value_counts())

# %% [markdown]
# ## 6. TREINAMENTO DOS 4 MODELOS

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Treinando os modelos...")

# 1. Random Forest
print("\n1. Treinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("   ✓ Random Forest treinado")

# 2. SVM
print("2. Treinando SVM...")
svm = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)
print("   ✓ SVM treinado")

# 3. KNN
print("3. Treinando KNN...")
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", KNeighborsClassifier(n_neighbors=5))
])
knn.fit(X_train, y_train)
print("   ✓ KNN treinado")

# 4. Gradient Boosting
print("4. Treinando Gradient Boosting...")
gb = GradientBoostingClassifier(
    random_state=42,
    n_estimators=100
)
gb.fit(X_train, y_train)
print("   ✓ Gradient Boosting treinado")

# %%
print("\nTodos os modelos foram treinados com sucesso!")

# %% [markdown]
# ## 7. PREDIÇÕES

# %%
print("Realizando predições...")

# Predições e probabilidades
pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]

pred_svm = svm.predict(X_test)
prob_svm = svm.predict_proba(X_test)[:, 1]

pred_knn = knn.predict(X_test)
prob_knn = knn.predict_proba(X_test)[:, 1]

pred_gb = gb.predict(X_test)
prob_gb = gb.predict_proba(X_test)[:, 1]

print("✓ Predições realizadas")

# %% [markdown]
# ## 8. CÁLCULO DE MÉTRICAS

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    specificity_score
)

def calcular_metricas(nome, y_real, y_pred, y_prob):
    """
    Calcula todas as métricas de classificação.
    """
    cm = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_real, y_pred),
        "precisao": precision_score(y_real, y_pred, zero_division=0),
        "recall_sensibilidade": recall_score(y_real, y_pred, zero_division=0),
        "especificidade": especificidade,
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_real, y_prob)
    }

# %%
print("Calculando métricas de cada modelo...\n")

resultados_modelos = []

resultados_modelos.append(
    calcular_metricas(
        "Random Forest",
        y_test,
        pred_rf,
        prob_rf
    )
)

resultados_modelos.append(
    calcular_metricas(
        "SVM",
        y_test,
        pred_svm,
        prob_svm
    )
)

resultados_modelos.append(
    calcular_metricas(
        "KNN",
        y_test,
        pred_knn,
        prob_knn
    )
)

resultados_modelos.append(
    calcular_metricas(
        "Gradient Boosting",
        y_test,
        pred_gb,
        prob_gb
    )
)

df_resultados = pd.DataFrame(resultados_modelos)

print("=" * 90)
print("RESULTADOS DOS MODELOS")
print("=" * 90)
print(df_resultados.to_string(index=False))
print("=" * 90)

# %% [markdown]
# ## 9. GRÁFICOS DE MATRIZES DE CONFUSÃO

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

modelos_info = [
    ("Random Forest", pred_rf, "Blues"),
    ("SVM", pred_svm, "Oranges"),
    ("KNN", pred_knn, "Greens"),
    ("Gradient Boosting", pred_gb, "Purples")
]

for idx, (nome, pred, cmap) in enumerate(modelos_info):
    cm = confusion_matrix(y_test, pred)
    
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    labels = np.array([
        [
            f"{cm[i,j]}\n({cm_percent[i,j]:.1%})"
            for j in range(cm.shape[1])
        ]
        for i in range(cm.shape[0])
    ])
    
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        cbar=False,
        ax=axes[idx]
    )
    
    axes[idx].set_title(f"Matriz de Confusão - {nome}", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Classe Predita")
    axes[idx].set_ylabel("Classe Real")

plt.tight_layout()
plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/01_matrizes_confusao.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de matrizes salvo: 01_matrizes_confusao.png")
plt.show()

# %% [markdown]
# ## 10. CURVAS ROC

# %%
from sklearn.metrics import roc_curve, auc

fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test, prob_svm)
fpr_knn, tpr_knn, _ = roc_curve(y_test, prob_knn)
fpr_gb, tpr_gb, _ = roc_curve(y_test, prob_gb)

auc_rf = auc(fpr_rf, tpr_rf)
auc_svm = auc(fpr_svm, tpr_svm)
auc_knn = auc(fpr_knn, tpr_knn)
auc_gb = auc(fpr_gb, tpr_gb)

plt.figure(figsize=(10, 8))

plt.plot(fpr_rf, tpr_rf, linewidth=3, label=f"Random Forest (AUC={auc_rf:.3f})", color='#1f77b4')
plt.plot(fpr_svm, tpr_svm, linewidth=3, label=f"SVM (AUC={auc_svm:.3f})", color='#ff7f0e')
plt.plot(fpr_knn, tpr_knn, linewidth=3, label=f"KNN (AUC={auc_knn:.3f})", color='#2ca02c')
plt.plot(fpr_gb, tpr_gb, linewidth=3, label=f"Gradient Boosting (AUC={auc_gb:.3f})", color='#d62728')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aleatório (AUC=0.5)')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=12)
plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=12)
plt.title("Curvas ROC - Comparação dos Modelos", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)

plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/02_curvas_roc.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de curvas ROC salvo: 02_curvas_roc.png")
plt.show()

# %% [markdown]
# ## 11. COMPARAÇÃO DE MÉTRICAS

# %%
metricas_plot = [
    "acuracia",
    "precisao",
    "recall_sensibilidade",
    "especificidade",
    "f1_score",
    "auc_roc"
]

fig, ax = plt.subplots(figsize=(14, 7))

df_plot = df_resultados.set_index("modelo")[metricas_plot]

df_plot.plot(kind="bar", ax=ax, width=0.8, colormap='Set2')

plt.title("Comparação de Métricas - Todos os Modelos", fontsize=14, fontweight='bold')
plt.ylabel("Valor da Métrica", fontsize=12)
plt.xlabel("Modelo", fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/03_comparacao_metricas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de comparação de métricas salvo: 03_comparacao_metricas.png")
plt.show()

# %% [markdown]
# ## 12. HEATMAP DE MÉTRICAS

# %%
fig, ax = plt.subplots(figsize=(10, 6))

df_heatmap = df_resultados.set_index("modelo")[metricas_plot]

sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Valor'},
    ax=ax,
    linewidths=0.5
)

ax.set_title("Heatmap de Métricas de Desempenho", fontsize=14, fontweight='bold')
ax.set_xlabel("Métricas", fontsize=12)
ax.set_ylabel("Modelos", fontsize=12)

plt.tight_layout()
plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/04_heatmap_metricas.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap de métricas salvo: 04_heatmap_metricas.png")
plt.show()

# %% [markdown]
# ## 13. GRÁFICOS ADICIONAIS

# %%
# Distribuição das classes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# No dataset completo
df_c4["rotulo"].value_counts().plot(kind="bar", ax=ax1, color=['#1f77b4', '#ff7f0e'])
ax1.set_title("Distribuição das Classes - Dataset Completo", fontsize=12, fontweight='bold')
ax1.set_xlabel("Classe")
ax1.set_ylabel("Quantidade")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

# Proporção
proporcoes = df_c4["rotulo"].value_counts(normalize=True)
ax2.pie(proporcoes.values, labels=proporcoes.index, autopct='%1.1f%%', 
        colors=['#1f77b4', '#ff7f0e'], startangle=90)
ax2.set_title("Proporção das Classes", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/05_distribuicao_classes.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de distribuição de classes salvo: 05_distribuicao_classes.png")
plt.show()

# %% [markdown]
# ## 14. ANÁLISE DE IMPORTÂNCIA DE FEATURES

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest
importancias_rf = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values()

importancias_rf.plot(kind="barh", ax=axes[0], color='#1f77b4')
axes[0].set_title("Importância das Features - Random Forest", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Importância")

# Gradient Boosting
importancias_gb = pd.Series(
    gb.feature_importances_,
    index=X.columns
).sort_values()

importancias_gb.plot(kind="barh", ax=axes[1], color='#d62728')
axes[1].set_title("Importância das Features - Gradient Boosting", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Importância")

plt.tight_layout()
plt.savefig('/home/ana/PycharmProjects/Visao_Computacional/presentation3/resultados/06_importancia_features.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de importância de features salvo: 06_importancia_features.png")
plt.show()

# %% [markdown]
# ## 15. RESUMO FINAL

# %%
print("\n" + "="*90)
print("RESUMO FINAL - CENÁRIO 4 COM 4 MODELOS")
print("="*90)
print(f"\nDataset:")
print(f"  • Total de nódulos: {len(df_c4)}")
print(f"  • Benignos: {len(df_c4[df_c4['rotulo'] == 'benigno'])}")
print(f"  • Malignos: {len(df_c4[df_c4['rotulo'] == 'maligno'])}")
print(f"  • Features por nódulo: {X.shape[1]}")

print(f"\nSplit Treino/Teste:")
print(f"  • Treino: {X_train.shape[0]} amostras (70%)")
print(f"  • Teste: {X_test.shape[0]} amostras (30%)")

print(f"\nMelhor Modelo: {df_resultados.loc[df_resultados['acuracia'].idxmax(), 'modelo']}")
print(f"  • Acurácia: {df_resultados['acuracia'].max():.4f}")
print(f"  • AUC-ROC: {df_resultados['auc_roc'].max():.4f}")

print(f"\nGráficos gerados:")
print(f"  ✓ 01_matrizes_confusao.png")
print(f"  ✓ 02_curvas_roc.png")
print(f"  ✓ 03_comparacao_metricas.png")
print(f"  ✓ 04_heatmap_metricas.png")
print(f"  ✓ 05_distribuicao_classes.png")
print(f"  ✓ 06_importancia_features.png")
print("="*90 + "\n")

# %%
print("Tabela final de resultados:")
print(df_resultados.to_string(index=False))
