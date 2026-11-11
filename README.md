# 💧⚡ Spatial Analysis of the Water-Energy Nexus in São Paulo

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)  
[![Geopandas](https://img.shields.io/badge/Geospatial-GeoPandas%20%2F%20PySal-green.svg)](https://geopandas.org/)  
[![Econometrics](https://img.shields.io/badge/Econometrics-SAR%20%2F%20SEM%20%2F%20GWR-orange.svg)](https://pysal.org/)  
[![Interactive Map](https://img.shields.io/badge/Dashboard-Folium%20%2F%20Leaflet-red.svg)](https://python-visualization.github.io/folium/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Visão Geral do Projeto

Este repositório apresenta uma pipeline analítica avançada de **Econometria Espacial e Geoprocessamento** aplicada ao **Nexo Água-Energia** nos municípios do Estado de São Paulo. O projeto investiga a relação espacial crítica entre as **perdas físicas de água nas redes de distribuição (%)** e o **consumo específico de energia elétrica (kWh/m³)** associado ao saneamento e bombeamento municipal.

O projeto integra técnicas modernas de Análise Exploratória de Dados Espaciais (ESDA), modelagem espacial global (**SAR/SEM**), regressão geograficamente ponderada (**GWR**) com validação cruzada espacial rigorosa (*Spatial Block CV*), e um **dashboard interativo web** em Folium.

---

## 🛠️ Arquitetura do Pipeline Analítico

O repositório é modularizado em 5 scripts sequenciais:

| Etapa | Script | Descrição Técnica |
| :--- | :--- | :--- |
| **01** | `01_geracao_datasets.py` | Geração, limpeza e consolidação da base de dados vetorial e tabular municipal. |
| **02** | `02_esda_moran_lisa.py` | Matriz de ponderação espacial (K-NN/Queen), Moran Global e LISA (*Hotspots* / *Coldspots*). |
| **03** | `03_modelos_econometricos_sar_sem.py` | Estimação de modelos espaciais globais (**SAR** - *Spatial Lag* e **SEM* - *Spatial Error*). |
| **04** | `04_gwr_validacao_cruzada.py` | Calibração de **GWR** e Validação Cruzada Espacial em Blocos (*Spatial Block 5-Fold CV*). |
| **05** | `05_mapas_interativos_web_folium.py` | Exportação de dashboard interativo em HTML utilizando tiles da *Esri World Street Map*. |

---

## 📊 Desempenho dos Modelos e Validação Cruzada

Resultados comparativos de desempenho preditivo (*In-Sample* vs *Out-of-Sample* via Spatial Block CV):

| Método / Modelo | R² (In-Sample) | RMSE (In-Sample) | R² (Spatial Block CV) | RMSE (Spatial Block CV) |
| :--- | :---: | :---: | :---: | :---: |
| **OLS Clássico** | 0.6615 | 0.0650 | 0.6185 | 0.0690 |
| **GWR Local** | **0.7349** | **0.0575** | 0.6068 | 0.0700 |

---

## 🌐 Dashboard Interativo (`outputs/`)

O projeto gera automaticamente um mapa interativo completo:
* **Arquivo:** `outputs/mapa_interativo_water_energy_sp.html`
* **Destaques:** Camadas dinâmicas selecionáveis para os Clusters LISA e Parâmetros Locais da GWR, tooltips com indicadores municipais e mapa base de alta resolução (*Esri World Street Map*).
---

## 🚀 Como Executar o Projeto

Clone o repositório e instale as dependências recomendadas (`geopandas`, `libpysal`, `esda`, `mgwr`, `spreg`, `folium`):

git clone [https://github.com/eniocancian/sp-water-energy-spatial-nexus.git](https://github.com/eniocancian/sp-water-energy-spatial-nexus.git)
cd sp-water-energy-spatial-nexus

👤 Autor
Ênio Fernando Cancian Júnior
