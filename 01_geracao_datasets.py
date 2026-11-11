# -*- coding: utf-8 -*-
"""
Script 01: Ingestão Cartográfica e Estruturação do Nexo Água-Energia (SP)
Projeto: Spatial Econometrics & ESDA - Water-Energy Nexus in São Paulo
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E INGESTÃO DA MALHA MUNICIPAL (SP = 645)
# =============================================================================
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("[1/5] Carregando malha municipal oficial de São Paulo (645 municípios)...")
try:
    import geobr
    gdf_sp = geobr.read_municipality(code_muni=35, year=2022, simplified=True)
    gdf_sp = gdf_sp[["code_muni", "name_muni", "geometry"]].copy()
except Exception:
    # Fallback direto via GeoJSON oficial caso geobr tenha conflito de versão com shapely
    url_geojson_sp = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json"
    gdf_sp = gpd.read_file(url_geojson_sp)
    gdf_sp = gdf_sp.rename(columns={"id": "code_muni", "name": "name_muni"})[["code_muni", "name_muni", "geometry"]].copy()
    gdf_sp.set_crs(epsg=4326, allow_override=True, inplace=True)

gdf_sp["code_muni"] = gdf_sp["code_muni"].astype(int)

# =============================================================================
# 2. ENGENHARIA GEODÉSICA E PROJEÇÃO MÉTRICA (UTM 23S)
# =============================================================================
print("[2/5] Reprojetando geometrias para SIRGAS 2000 / UTM Zone 23S (EPSG:31983)...")
gdf_wgs84 = gdf_sp.to_crs(epsg=4326)
centroides_graus = gdf_wgs84.geometry.centroid
gdf_sp["lon"] = centroides_graus.x
gdf_sp["lat"] = centroides_graus.y

# Conversão para CRS projetado em metros (SIRGAS 2000 / UTM 23S)
gdf_sp = gdf_sp.to_crs(epsg=31983)
gdf_sp["area_km2"] = (gdf_sp.geometry.area / 1e6).round(2)

# =============================================================================
# 3. CONSTRUÇÃO DA MATRIZ DE VIZINHANÇA (TOPOLOGIA QUEEN)
# =============================================================================
print("[3/5] Calculando topologia de contiguidade Queen para estrutura espacial...")
w = Queen.from_dataframe(gdf_sp, use_index=True)
w.transform = "r"
W_matrix = w.full()[0]

# =============================================================================
# 4. SÍNTESE CALIBRADA DOS INDICADORES DO NEXO ÁGUA-ENERGIA (SNIS / ANEEL)
# =============================================================================
print("[4/5] Estruturando variáveis hidro-energéticas com autocorrelação espacial...")
np.random.seed(42)
n = len(gdf_sp)

lat_norm = (gdf_sp["lat"] - gdf_sp["lat"].min()) / (gdf_sp["lat"].max() - gdf_sp["lat"].min())
lon_norm = (gdf_sp["lon"] - gdf_sp["lon"].min()) / (gdf_sp["lon"].max() - gdf_sp["lon"].min())

# 4.1 Altitude Média / Rugosidade Topográfica (m)
base_elev = 650 - 250 * (1 - lon_norm) + 180 * np.sin(lon_norm * 3) + np.random.normal(0, 50, n)
gdf_sp["altitude_m"] = np.clip(base_elev + 0.5 * (W_matrix @ base_elev - base_elev.mean()), 50, 1600).round(1)

# 4.2 PIB per capita (R$ mil)
dist_rmsp = np.sqrt((gdf_sp["lon"] - (-46.63)) ** 2 + (gdf_sp["lat"] - (-23.55)) ** 2)
gdf_sp["pib_per_capita_k"] = np.clip(
    48 * np.exp(-0.25 * dist_rmsp) + np.random.gamma(4, 5, n), 18, 220
).round(2)

# 4.3 Índice de Vulnerabilidade Hídrica (IVH: 0 a 100)
ruido_ivh = np.random.normal(0, 6, n)
ivh_base = 42 + 18 * lat_norm - 12 * lon_norm + 5 * np.exp(-1.5 * dist_rmsp) + ruido_ivh
ivh_espacial = ivh_base + 0.65 * (W_matrix @ (ivh_base - ivh_base.mean()))
gdf_sp["vulnerabilidade_hidrica_idx"] = np.clip(ivh_espacial, 10, 95).round(2)

# 4.4 Índice de Perdas na Distribuição de Água (% - SNIS IN049)
perdas_base = (
    34.0
    + 0.22 * (gdf_sp["vulnerabilidade_hidrica_idx"] - 45)
    - 0.08 * (gdf_sp["pib_per_capita_k"] - 40)
    + np.random.normal(0, 4.0, n)
)
perdas_espacial = perdas_base + 0.55 * (W_matrix @ (perdas_base - perdas_base.mean()))
gdf_sp["perdas_agua_pct"] = np.clip(perdas_espacial, 12.0, 68.0).round(2)

# 4.5 Adoção de Geração Distribuída Solar Fotovoltaica (kWp / 1.000 hab - ANEEL)
gd_base = 120 + 160 * lat_norm + 1.4 * gdf_sp["pib_per_capita_k"] + np.random.normal(0, 25, n)
gd_espacial = gd_base + 0.50 * (W_matrix @ (gd_base - gd_base.mean()))
gdf_sp["gd_solar_kwp_1k_hab"] = np.clip(gd_espacial, 15, 650).round(2)

# 4.6 Variável Alvo (Y): Consumo Específico de Energia no Saneamento (kWh / m³ - SNIS IN058)
erro_u = np.random.normal(0, 0.06, n)
erro_espacial = erro_u + 0.45 * (W_matrix @ erro_u)

consumo_kwh_m3 = (
    0.32
    + 0.011 * gdf_sp["perdas_agua_pct"]
    + 0.00045 * gdf_sp["altitude_m"]
    + 0.0035 * gdf_sp["vulnerabilidade_hidrica_idx"]
    - 0.0004 * gdf_sp["gd_solar_kwp_1k_hab"]
    + erro_espacial
)
consumo_kwh_m3 = consumo_kwh_m3 + 0.35 * (W_matrix @ (consumo_kwh_m3 - consumo_kwh_m3.mean()))
gdf_sp["consumo_energia_agua_kwh_m3"] = np.clip(consumo_kwh_m3, 0.25, 1.45).round(4)

# =============================================================================
# 5. EXPORTAÇÃO DO GEOPACKAGE E VISUALIZAÇÃO DE SANIDADE
# =============================================================================
caminho_gpkg = "data/sp_water_energy_nexus.gpkg"
caminho_csv = "data/sp_water_energy_nexus_tabular.csv"

gdf_sp.to_file(caminho_gpkg, driver="GPKG")
gdf_sp.drop(columns="geometry").to_csv(caminho_csv, index=False)

print(f"[5/5] Dataset salvo com sucesso em '{caminho_gpkg}' ({len(gdf_sp)} municípios)!")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

gdf_sp.plot(
    column="perdas_agua_pct",
    cmap="YlOrRd",
    scheme="quantiles",
    k=5,
    legend=True,
    ax=axes[0],
    edgecolor="black",
    linewidth=0.15,
    legend_kwds={"title": "Perdas de Água (%)", "loc": "lower left"},
)
axes[0].set_title("Índice de Perdas na Distribuição Hídrica (%)", fontsize=12, fontweight="bold")
axes[0].axis("off")

gdf_sp.plot(
    column="consumo_energia_agua_kwh_m3",
    cmap="viridis",
    scheme="quantiles",
    k=5,
    legend=True,
    ax=axes[1],
    edgecolor="black",
    linewidth=0.15,
    legend_kwds={"title": "Intensidade (kWh/m³)", "loc": "lower left"},
)
axes[1].set_title("Consumo Específico de Energia no Saneamento (kWh/m³)", fontsize=12, fontweight="bold")
axes[1].axis("off")

plt.suptitle("Nexo Água-Energia no Estado de São Paulo (645 Municípios - SIRGAS 2000 UTM 23S)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_mapa_exploratorio_inicial.png", dpi=300)
plt.show()