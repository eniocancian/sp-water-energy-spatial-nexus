# -*- coding: utf-8 -*-
"""
Script 04: Regressão Ponderada Geograficamente (GWR) e Validação Cruzada Espacial
Dica no Spyder: Pressione F5 para rodar tudo ou Ctrl + Enter célula por célula (na ordem 1 -> 5)
"""

# %% [CÉLULA 1] Importações, Projeção Métrica (UTM) e Preparação das Matrizes
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

print("[Célula 1] Carregando GeoPackage e projetando coordenadas para UTM (SIRGAS 2000 / 23S)...")
gdf = gpd.read_file("data/sp_water_energy_nexus.gpkg")

# Projeta para EPSG:31983 (metros) para cálculo exato de distâncias geográficas em km
gdf_utm = gdf.to_crs(epsg=31983)
coords_km = np.column_stack([gdf_utm.geometry.centroid.x / 1000.0, gdf_utm.geometry.centroid.y / 1000.0])

y_name = "consumo_energia_agua_kwh_m3"
x_names = [
    "perdas_agua_pct",
    "altitude_m",
    "vulnerabilidade_hidrica_idx",
    "gd_solar_kwp_1k_hab",
    "pib_per_capita_k"
]

y = gdf[y_name].values
X_raw = gdf[x_names].values
# Adiciona coluna de 1s (Intercepto)
X = np.column_stack([np.ones(len(gdf)), X_raw])

# Matriz de distância euclidiana entre todos os 645 municípios (em km)
dist_matrix = cdist(coords_km, coords_km, metric="euclidean")
print(f"Matriz de Distâncias calculada: {dist_matrix.shape} | Distância máxima em SP: {dist_matrix.max():.1f} km")


# %% [CÉLULA 2] Calibração da Largura de Banda Ótima (Adaptive Bisquare Kernel)
print("[Célula 2] Otimizando largura de banda adaptativa (k-vizinhos mais próximos via AICc)...")

def fit_gwr_adaptive(X, y, dist_mat, k_neighbors):
    n, p = X.shape
    betas = np.zeros((n, p))
    y_hat = np.zeros(n)
    hat_diag = np.zeros(n)
    local_r2 = np.zeros(n)
    
    for i in range(n):
        dists = dist_mat[i]
        # Distância até o k-ésimo vizinho mais próximo define o raio local (bw)
        bw = np.sort(dists)[k_neighbors]
        
        # Kernel Bisquare: w_ij = (1 - (d_ij / bw)^2)^2 para d_ij < bw
        w = np.where(dists < bw, (1.0 - (dists / bw) ** 2) ** 2, 0.0)
        
        # Mínimos Quadrados Ponderados Localmente: beta_i = (X' W_i X)^(-1) X' W_i y
        XW = X.T * w
        XtWX_inv = np.linalg.pinv(XW @ X)
        beta_i = XtWX_inv @ (XW @ y)
        
        betas[i] = beta_i
        y_hat[i] = X[i] @ beta_i
        hat_diag[i] = X[i] @ XtWX_inv @ XW[:, i]
        
        # R² Local ponderado pelo kernel
        y_bar_w = np.sum(w * y) / np.sum(w)
        tss_w = np.sum(w * (y - y_bar_w) ** 2)
        rss_w = np.sum(w * (y - (X @ beta_i)) ** 2)
        local_r2[i] = 1.0 - (rss_w / tss_w)
        
    residuals = y - y_hat
    rss = np.sum(residuals ** 2)
    tr_S = np.sum(hat_diag)  # Número efetivo de parâmetros (graus de liberdade gastos)
    sigma2 = rss / n
    
    # Critério de Informação de Akaike Corrigido (AICc de Fotheringham et al.)
    aicc = 2.0 * n * np.log(np.sqrt(sigma2)) + n * np.log(2.0 * np.pi) + n * ((n + tr_S) / (n - 2.0 - tr_S))
    r2_global = r2_score(y, y_hat)
    
    return betas, y_hat, local_r2, aicc, tr_S, r2_global

# Testa larguras de banda adaptativas de 45 a 150 municípios vizinhos
candidatos_k = [45, 60, 75, 90, 110, 130, 150]
historico_bw = []

for k in candidatos_k:
    _, _, _, aicc_val, tr_s_val, r2_val = fit_gwr_adaptive(X, y, dist_matrix, k)
    historico_bw.append({"Vizinhos_k": k, "Parametros_Efetivos": round(tr_s_val, 1), "R2_Global": round(r2_val, 4), "AICc": round(aicc_val, 2)})

df_bw = pd.DataFrame(historico_bw)
best_k = int(df_bw.loc[df_bw["AICc"].idxmin(), "Vizinhos_k"])

print("\n=== BUSCA DE LARGURA DE BANDA GWR (CRITÉRIO AICc) ===")
print(df_bw.to_string(index=False))
print(f"\n-> Largura de Banda Ótima Selecionada: k = {best_k} municípios vizinhos")


# %% [CÉLULA 3] Ajuste Final do Modelo GWR e Extração dos Coeficientes Locais
print(f"[Célula 3] Estimando modelo GWR final com k = {best_k} vizinhos...")

betas_gwr, y_hat_gwr, local_r2_gwr, aicc_gwr, tr_s_gwr, r2_gwr = fit_gwr_adaptive(X, y, dist_matrix, best_k)

# Salva as colunas no GeoDataFrame (garantindo que a Célula 5 funcione perfeitamente!)
gdf["gwr_intercept"] = betas_gwr[:, 0]
gdf["gwr_beta_perdas"] = betas_gwr[:, 1]
gdf["gwr_beta_altitude"] = betas_gwr[:, 2]
gdf["gwr_beta_vulnerab"] = betas_gwr[:, 3]
gdf["gwr_beta_solar"] = betas_gwr[:, 4]
gdf["gwr_beta_pib"] = betas_gwr[:, 5]
gdf["gwr_local_r2"] = local_r2_gwr
gdf["pred_gwr_kwh_m3"] = y_hat_gwr

print("\n>>> RESUMO DO MODELO GWR (REGRESSÃO PONDERADA GEOGRAFICAMENTE) <<<")
print(f"R² Global GWR         : {r2_gwr:.4f} (vs. 0.6615 no OLS)")
print(f"R² Local Médio        : {local_r2_gwr.mean():.4f} (Min: {local_r2_gwr.min():.4f} | Max: {local_r2_gwr.max():.4f})")
print(f"Parâmetros Efetivos   : {tr_s_gwr:.2f}")
print(f"AICc GWR              : {aicc_gwr:.2f}")

resumo_betas = pd.DataFrame(
    betas_gwr,
    columns=["Intercepto", "perdas_agua_pct", "altitude_m", "vulnerabilidade_idx", "gd_solar_kwp", "pib_per_capita_k"]
).describe().T[["mean", "std", "min", "50%", "max"]].round(5)

print("\n=== DISTRIBUIÇÃO DOS PARÂMETROS LOCAIS GWR NOS 645 MUNICÍPIOS ===")
print(resumo_betas.to_string())


# %% [CÉLULA 4] Validação Cruzada Espacial em Blocos (Spatial Block 5-Fold CV)
print("\n[Célula 4] Executando Validação Cruzada Espacial em Blocos (5 Regiões Geográficas)...")

# Divide o Estado de São Paulo em 5 macrorregiões espaciais via K-Means nas coordenadas UTM
kmeans_blocks = KMeans(n_clusters=5, random_state=42, n_init=10).fit(coords_km)
spatial_folds = kmeans_blocks.labels_
gdf["spatial_cv_fold"] = spatial_folds

preds_ols_cv = np.zeros(len(gdf))
preds_gwr_cv = np.zeros(len(gdf))

for fold in range(5):
    test_idx = (spatial_folds == fold)
    train_idx = ~test_idx
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 1. Predição OLS Out-of-Sample
    beta_ols_fold = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ y_train)
    preds_ols_cv[test_idx] = X_test @ beta_ols_fold
    
    # 2. Predição GWR Out-of-Sample (ponderando pelos vizinhos do conjunto de treino)
    dist_test_train = cdist(coords_km[test_idx], coords_km[train_idx], metric="euclidean")
    for idx_local, i_global in enumerate(np.where(test_idx)[0]):
        d_vec = dist_test_train[idx_local]
        bw_cv = np.sort(d_vec)[min(best_k, len(y_train) - 1)]
        w_cv = np.where(d_vec < bw_cv, (1.0 - (d_vec / bw_cv) ** 2) ** 2, 0.0)
        
        XW_cv = X_train.T * w_cv
        beta_i_cv = np.linalg.pinv(XW_cv @ X_train) @ (XW_cv @ y_train)
        preds_gwr_cv[i_global] = X[i_global] @ beta_i_cv

df_cv = pd.DataFrame({
    "Método de Validação": [
        "OLS Clássico (In-Sample Treino)",
        "GWR Local (In-Sample Treino)",
        "OLS - Spatial Block 5-Fold CV (Out-of-Sample)",
        "GWR - Spatial Block 5-Fold CV (Out-of-Sample)"
    ],
    "RMSE (kWh/m³)": [
        np.sqrt(mean_squared_error(y, X @ (np.linalg.pinv(X.T @ X) @ (X.T @ y)))),
        np.sqrt(mean_squared_error(y, y_hat_gwr)),
        np.sqrt(mean_squared_error(y, preds_ols_cv)),
        np.sqrt(mean_squared_error(y, preds_gwr_cv))
    ],
    "MAE (kWh/m³)": [
        mean_absolute_error(y, X @ (np.linalg.pinv(X.T @ X) @ (X.T @ y))),
        mean_absolute_error(y, y_hat_gwr),
        mean_absolute_error(y, preds_ols_cv),
        mean_absolute_error(y, preds_gwr_cv)
    ],
    "R²": [
        r2_score(y, X @ (np.linalg.pinv(X.T @ X) @ (X.T @ y))),
        r2_gwr,
        r2_score(y, preds_ols_cv),
        r2_score(y, preds_gwr_cv)
    ]
}).round(4)

print("\n=== RESULTADOS DA VALIDAÇÃO CRUZADA ESPACIAL (SPATIAL BLOCK CV) ===")
print(df_cv.to_string(index=False))
df_cv.to_csv("outputs/04_validacao_cruzada_espacial_gwr.csv", index=False)


# %% [CÉLULA 5] Visualização da Variação Espacial dos Parâmetros Locais e R² Local
print("\n[Célula 5] Mapeando a variação regional dos coeficientes da GWR...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

# Mapa 1: Coeficiente Local de Perdas de Água
gdf.plot(
    column="gwr_beta_perdas", cmap="OrRd", scheme="quantiles", k=5,
    legend=True, ax=axes[0], edgecolor="black", linewidth=0.15,
    legend_kwds={"title": "β Local (Perdas)", "loc": "lower left", "fmt": "{:.4f}"}
)
axes[0].set_title("Elasticidade Local: Perdas de Água (β₁)\n(Onde o desperdício hídrico encarece mais a energia)", fontsize=11, fontweight="bold")
axes[0].axis("off")

# Mapa 2: Coeficiente Local de Geração Solar
gdf.plot(
    column="gwr_beta_solar", cmap="BuGn_r", scheme="quantiles", k=5,
    legend=True, ax=axes[1], edgecolor="black", linewidth=0.15,
    legend_kwds={"title": "β Local (Solar)", "loc": "lower left", "fmt": "{:.5f}"}
)
axes[1].set_title("Mitigação Local: Geração Solar GD (β₄)\n(Valores mais negativos = Maior alívio energético)", fontsize=11, fontweight="bold")
axes[1].axis("off")

# Mapa 3: R² Local da GWR
gdf.plot(
    column="gwr_local_r2", cmap="YlGnBu", scheme="quantiles", k=5,
    legend=True, ax=axes[2], edgecolor="black", linewidth=0.15,
    legend_kwds={"title": "R² Local", "loc": "lower left", "fmt": "{:.3f}"}
)
axes[2].set_title(f"Poder Explicativo Regional (R² Local GWR)\nR² Global = {r2_gwr:.3f} (k = {best_k} vizinhos)", fontsize=11, fontweight="bold")
axes[2].axis("off")

plt.suptitle("Modelagem GWR: Heterogeneidade Espacial dos Parâmetros no Estado de São Paulo", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_mapa_parametros_locais_gwr.png", dpi=300)
plt.show()

# Salva o GeoPackage final com todas as colunas de OLS, SAR, SEM e GWR
gdf.to_file("data/sp_water_energy_nexus.gpkg", driver="GPKG")
print("\nScript 04 concluído com sucesso! Mapa salvo em 'outputs/04_mapa_parametros_locais_gwr.png'.")