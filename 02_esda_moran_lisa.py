# -*- coding: utf-8 -*-
"""
Script 02: Análise Exploratória de Dados Espaciais (ESDA) - Moran Global e LISA Bivariado
Dica no Spyder: Execute célula por célula pressionando Ctrl + Enter
"""

# %% [CÉLULA 1] Importações, Leitura do GeoPackage e Matriz Queen (W)
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from libpysal.weights import Queen
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV

warnings.filterwarnings("ignore")

print("[Célula 1] Carregando GeoPackage e construindo matriz de contiguidade Queen (W)...")
gdf = gpd.read_file("data/sp_water_energy_nexus.gpkg")

# Matriz de vizinhança Queen padronizada por linha (Row-standardized)
w = Queen.from_dataframe(gdf, use_index=True)
w.transform = "r"
print(f"Malha carregada: {len(gdf)} municípios | Média de vizinhos por município: {w.mean_neighbors:.2f}")


# %% [CÉLULA 2] Índice de Moran Global (Univariado e Bivariado)
print("[Célula 2] Calculando Índices de Moran Global (999 permutações Monte Carlo)...")
np.random.seed(42)

y_energia = gdf["consumo_energia_agua_kwh_m3"].values
x_perdas = gdf["perdas_agua_pct"].values
x_ivh = gdf["vulnerabilidade_hidrica_idx"].values
x_solar = gdf["gd_solar_kwp_1k_hab"].values

# Testes Globais Univariados
mi_energia = Moran(y_energia, w, permutations=999)
mi_perdas = Moran(x_perdas, w, permutations=999)
mi_ivh = Moran(x_ivh, w, permutations=999)
mi_solar = Moran(x_solar, w, permutations=999)

# Teste Global Bivariado: Perdas de Água (X) vs Consumo Energético dos Vizinhos (Wy)
mi_bv_perdas_energia = Moran_BV(x_perdas, y_energia, w, permutations=999)

df_moran = pd.DataFrame({
    "Indicador / Relação": [
        "Consumo Específico Energia (kWh/m³)",
        "Perdas na Distribuição de Água (%)",
        "Índice de Vulnerabilidade Hídrica (IVH)",
        "Geração Distribuída Solar (kWp/1k hab)",
        "Bivariado: Perdas Água (X) -> Energia Vizinhos (Wy)"
    ],
    "Moran_I": [
        mi_energia.I, mi_perdas.I, mi_ivh.I, mi_solar.I, mi_bv_perdas_energia.I
    ],
    "Z_Score": [
        mi_energia.z_sim, mi_perdas.z_sim, mi_ivh.z_sim, mi_solar.z_sim, mi_bv_perdas_energia.z_sim
    ],
    "p_value_sim": [
        mi_energia.p_sim, mi_perdas.p_sim, mi_ivh.p_sim, mi_solar.p_sim, mi_bv_perdas_energia.p_sim
    ]
}).round(4)

print("\n=== RESULTADOS DO TESTE DE MORAN GLOBAL ===")
print(df_moran.to_string(index=False))
df_moran.to_csv("outputs/02_tabela_moran_global.csv", index=False)


# %% [CÉLULA 3] Diagramas de Dispersão de Moran (Moran Scatterplots)
print("[Célula 3] Gerando Diagramas de Dispersão de Moran...")

z_energia = (y_energia - y_energia.mean()) / y_energia.std()
wz_energia = w.sparse @ z_energia
z_perdas = (x_perdas - x_perdas.mean()) / x_perdas.std()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Univariado (Consumo de Energia)
axes[0].scatter(z_energia, wz_energia, alpha=0.5, c="#1f77b4", edgecolors="k", linewidth=0.3, s=30)
axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
linha_x = np.linspace(z_energia.min(), z_energia.max(), 100)
axes[0].plot(linha_x, mi_energia.I * linha_x, color="#d62728", linewidth=2.2,
             label=f"Moran's I = {mi_energia.I:.3f} (p = {mi_energia.p_sim:.3f})")
axes[0].set_title("Moran Scatterplot Univariado\nConsumo Energético no Saneamento (kWh/m³)", fontweight="bold")
axes[0].set_xlabel("Consumo Energético Padronizado (Z)")
axes[0].set_ylabel("Lag Espacial do Consumo Energético (WZ)")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.25)

# Subplot 2: Bivariado (Perdas de Água X vs Lag Energia WZ)
axes[1].scatter(z_perdas, wz_energia, alpha=0.5, c="#ff7f0e", edgecolors="k", linewidth=0.3, s=30)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
linha_bv = np.linspace(z_perdas.min(), z_perdas.max(), 100)
axes[1].plot(linha_bv, mi_bv_perdas_energia.I * linha_bv, color="#d62728", linewidth=2.2,
             label=f"Bivariate Moran's I = {mi_bv_perdas_energia.I:.3f} (p = {mi_bv_perdas_energia.p_sim:.3f})")
axes[1].set_title("Moran Scatterplot Bivariado (Nexo Água-Energia)\nPerdas de Água (X) vs. Lag Consumo Energético (WY)", fontweight="bold")
axes[1].set_xlabel("Índice de Perdas de Água Padronizado (Z_x)")
axes[1].set_ylabel("Lag Espacial do Consumo Energético (W Z_y)")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("outputs/02_moran_scatterplots.png", dpi=300)
plt.show()


# %% [CÉLULA 4] Cálculo dos Clusters Locais LISA (Univariado e Bivariado)
print("[Célula 4] Calculando clusters locais LISA (Hotspots e Coldspots)...")
lisa_energia = Moran_Local(y_energia, w, permutations=999, seed=42)
lisa_bv = Moran_Local_BV(x_perdas, y_energia, w, permutations=999, seed=42)

mapa_labels = {
    0: "Não Significativo (p >= 0.05)",
    1: "High-High (Hotspot Crítico)",
    2: "Low-High (Outlier Espacial)",
    3: "Low-Low (Coldspot Eficiente)",
    4: "High-Low (Outlier Isolado)"
}

def classificar_lisa(lisa_obj, alpha=0.05):
    quadrantes = np.where(lisa_obj.p_sim < alpha, lisa_obj.q, 0)
    return [mapa_labels[q] for q in quadrantes]

gdf["lisa_energia_cluster"] = classificar_lisa(lisa_energia, alpha=0.05)
gdf["lisa_bv_nexus_cluster"] = classificar_lisa(lisa_bv, alpha=0.05)

print("\nContagem de Municípios por Cluster - LISA Bivariado (Perdas Água -> Energia):")
print(gdf["lisa_bv_nexus_cluster"].value_counts())


# %% [CÉLULA 5] Plotagem dos Mapas LISA e Exportação do GeoPackage Atualizado
print("[Célula 5] Renderizando mapas coropléticos LISA e salvando GeoPackage...")

cores_lisa = {
    "High-High (Hotspot Crítico)": "#d7191c",       # Vermelho forte
    "Low-Low (Coldspot Eficiente)": "#2c7bb6",      # Azul forte
    "Low-High (Outlier Espacial)": "#abd9e9",       # Azul claro
    "High-Low (Outlier Isolado)": "#fdae61",        # Laranja claro
    "Não Significativo (p >= 0.05)": "#eeeeee"      # Cinza neutro
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Mapa 1: LISA Univariado
gdf.plot(ax=axes[0], color=gdf["lisa_energia_cluster"].map(cores_lisa), edgecolor="black", linewidth=0.15)
axes[0].set_title("LISA Univariado: Clusters de Intensidade Energética (kWh/m³)", fontsize=12, fontweight="bold")
axes[0].axis("off")

# Mapa 2: LISA Bivariado
gdf.plot(ax=axes[1], color=gdf["lisa_bv_nexus_cluster"].map(cores_lisa), edgecolor="black", linewidth=0.15)
axes[1].set_title("LISA Bivariado: Perdas Hídricas (X) x Intensidade Energética Vizinha (WY)", fontsize=12, fontweight="bold")
axes[1].axis("off")

patches = [mpatches.Patch(color=cor, label=label) for label, cor in cores_lisa.items()]
fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True, fontsize=10,
           title="Classificação de Clusters Espaciais LISA (α = 5%)")

plt.suptitle("Análise de Associação Espacial Local (LISA) - Nexo Água-Energia em São Paulo", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig("outputs/02_mapa_clusters_lisa_bivariado.png", dpi=300)
plt.show()

# Salva o GeoPackage enriquecido com as colunas dos clusters LISA para o Script 03
gdf.to_file("data/sp_water_energy_nexus.gpkg", driver="GPKG")
print("Script 02 concluído! GeoPackage atualizado e figuras salvas em 'outputs/'.")