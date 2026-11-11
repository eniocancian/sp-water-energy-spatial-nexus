# -*- coding: utf-8 -*-
"""
Script 03: Econometria Espacial (OLS vs. Spatial Lag SAR vs. Spatial Error SEM)
Dica no Spyder: Execute célula por célula pressionando Ctrl + Enter
"""

# %% [CÉLULA 1] Importações, Leitura do GeoPackage e Preparação das Matrizes (Y e X)
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran
import spreg

warnings.filterwarnings("ignore")

print("[Célula 1] Carregando GeoPackage enriquecido e preparando matrizes...")
gdf = gpd.read_file("data/sp_water_energy_nexus.gpkg")

# Matriz de contiguidade Queen padronizada por linha
w = Queen.from_dataframe(gdf, use_index=True)
w.transform = "r"

# Definição da Variável Dependente (Y) e Explicativas (X)
y_name = "consumo_energia_agua_kwh_m3"
x_names = [
    "perdas_agua_pct",
    "altitude_m",
    "vulnerabilidade_hidrica_idx",
    "gd_solar_kwp_1k_hab",
    "pib_per_capita_k"
]

y = gdf[[y_name]].values
X = gdf[x_names].values

print(f"Matriz Y: {y.shape} | Matriz X: {X.shape}")


# %% [CÉLULA 2] Modelo Baseline: Regressão Linear Clássica (OLS) com Diagnóstico Espacial
print("[Célula 2] Ajustando OLS Clássico e Testes de Multiplicadores de Lagrange (LM)...")

# O spreg.OLS já roda automaticamente o Moran's I dos resíduos e os testes LM (Lag e Error)
ols_model = spreg.OLS(
    y, X, w=w,
    spat_diag=True,
    moran=True,
    name_y=y_name,
    name_x=x_names,
    name_w="Queen_Contiguity",
    name_ds="SP_Water_Energy_Nexus"
)

print(ols_model.summary)


# %% [CÉLULA 3] Modelos Econométricos Espaciais: Spatial Lag (SAR) e Spatial Error (SEM)
print("[Célula 3] Estimando Modelos Espaciais via Máxima Verossimilhança (ML)...")

# 1. Modelo Spatial Lag (SAR): inclui a defasagem espacial da variável alvo (rho * W_y)
sar_model = spreg.ML_Lag(
    y, X, w=w,
    name_y=y_name,
    name_x=x_names,
    name_w="Queen_Contiguity",
    name_ds="SP_Water_Energy_Nexus"
)

# 2. Modelo Spatial Error (SEM): inclui a dependência espacial no termo de erro (lambda * W_u)
sem_model = spreg.ML_Error(
    y, X, w=w,
    name_y=y_name,
    name_x=x_names,
    name_w="Queen_Contiguity",
    name_ds="SP_Water_Energy_Nexus"
)

print("\n>>> RESUMO DO MODELO SPATIAL LAG (SAR) <<<")
print(sar_model.summary)

print("\n>>> RESUMO DO MODELO SPATIAL ERROR (SEM) <<<")
print(sem_model.summary)


# %% [CÉLULA 4] Comparação de Performance (AIC, Log-Likelihood e Moran nos Resíduos)
print("[Célula 4] Comparando métricas de ajuste e autocorrelação residual...")
np.random.seed(42)

# Cálculo direto do Índice de Moran sobre os resíduos de cada modelo
res_ols = ols_model.u.flatten()
res_sar = sar_model.u.flatten()
res_sem = sem_model.e_filtered.flatten()  # Corrigido: e_filtered (com underline)

mi_res_ols = Moran(res_ols, w, permutations=999)
mi_res_sar = Moran(res_sar, w, permutations=999)
mi_res_sem = Moran(res_sem, w, permutations=999)

df_comp = pd.DataFrame({
    "Modelo": [
        "1. OLS Clássico (Não-Espacial)",
        "2. Spatial Lag - SAR (ML_Lag)",
        "3. Spatial Error - SEM (ML_Error)"
    ],
    "R2_ou_PseudoR2": [
        ols_model.r2,
        sar_model.pr2,
        sem_model.pr2
    ],
    "Log_Likelihood": [
        ols_model.logll,
        sar_model.logll,
        sem_model.logll
    ],
    "AIC (Menor = Melhor)": [
        ols_model.aic,
        sar_model.aic,
        sem_model.aic
    ],
    "Param_Espacial (rho/lambda)": [
        0.0,
        float(np.squeeze(sar_model.rho)),
        float(np.squeeze(sem_model.lam))
    ],
    "Moran_I_Residuos": [
        mi_res_ols.I,
        mi_res_sar.I,
        mi_res_sem.I
    ],
    "p_value_Residuos": [
        mi_res_ols.p_sim,
        mi_res_sar.p_sim,
        mi_res_sem.p_sim
    ]
}).round(4)

print("\n=== TABELA COMPARATIVA DE MODELOS ECONOMÉTRICOS ===")
print(df_comp.to_string(index=False))
df_comp.to_csv("outputs/03_comparativo_modelos_espaciais.csv", index=False)

# %% [CÉLULA 5] Visualização: Mapa de Resíduos OLS vs. Resíduos Espaciais Filtrados
print("[Célula 5] Gerando mapa comparativo de resíduos e salvando predições...")

gdf["residuos_ols"] = res_ols
gdf["residuos_sar"] = res_sar
gdf["residuos_sem_filtrados"] = res_sem
gdf["pred_sar_kwh_m3"] = sar_model.predy.flatten().round(4)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

gdf.plot(
    column="residuos_ols",
    cmap="coolwarm",
    scheme="quantiles",
    k=5,
    legend=True,
    ax=axes[0],
    edgecolor="black",
    linewidth=0.15,
    legend_kwds={"title": "Resíduo OLS", "loc": "lower left"}
)
axes[0].set_title(
    f"Resíduos do Modelo OLS Clássico\nMoran's I = {mi_res_ols.I:.3f} (p = {mi_res_ols.p_sim:.3f} -> Viés Espacial!)",
    fontsize=11.5, fontweight="bold"
)
axes[0].axis("off")

gdf.plot(
    column="residuos_sem_filtrados",
    cmap="coolwarm",
    scheme="quantiles",
    k=5,
    legend=True,
    ax=axes[1],
    edgecolor="black",
    linewidth=0.15,
    legend_kwds={"title": "Resíduo Filtrado", "loc": "lower left"}
)
axes[1].set_title(
    f"Resíduos após Correção Espacial (SEM / SAR)\nMoran's I = {mi_res_sem.I:.3f} (p = {mi_res_sem.p_sim:.3f} -> Ruído Branco!)",
    fontsize=11.5, fontweight="bold"
)
axes[1].axis("off")

plt.suptitle("Diagnóstico de Resíduos: Regressão Clássica (OLS) vs. Econometria Espacial", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_mapa_residuos_ols_vs_espacial.png", dpi=300)
plt.show()

gdf.to_file("data/sp_water_energy_nexus.gpkg", driver="GPKG")


# %% [CÉLULA 6] Estudo de Caso: Presidente Prudente vs. São Paulo
print("[Célula 6] Consultando diagnóstico específico: Presidente Prudente vs. São Paulo...")

cidades_foco = gdf[gdf["name_muni"].isin(["Presidente Prudente", "São Paulo"])].copy()

colunas_foco = [
    "name_muni",
    "perdas_agua_pct",
    "vulnerabilidade_hidrica_idx",
    "gd_solar_kwp_1k_hab",
    "consumo_energia_agua_kwh_m3",
    "pred_sar_kwh_m3",
    "lisa_energia_cluster",
    "lisa_bv_nexus_cluster"
]

print("\n=== RAIO-X: PRESIDENTE PRUDENTE VS. SÃO PAULO ===")
print(cidades_foco[colunas_foco].to_string(index=False))