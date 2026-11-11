# Análise Espacial do Nexo Água-Energia no Estado de São Paulo

Repositório analítico desenvolvido para modelagem econométrica espacial, exploração estatística de dados espaciais (ESDA), modelagem espacial avançada (SAR/SEM), regressão geograficamente ponderada (GWR) com validação cruzada espacial e visualização cartográfica interativa aplicada aos municípios do Estado de São Paulo.

---

## 📋 Visão Geral do Projeto
Este projeto investiga o **Nexo Água-Energia**, mapeando a relação crítica entre as **perdas físicas de água (%)** nas redes de distribuição e o **consumo específico de energia elétrica (kWh/m³)** associado aos sistemas de saneamento básico e bombeamento municipal.

Utilizando o ecossistema Python em geoprocessamento e econometria espacial, o pipeline integra bases de dados geográficas municipais e executa modelagens avançadas para capturar dependências espaciais e heterogeneidades regionais.

---

## 🛠️ Arquitetura do Pipeline Analítico
O repositório é estruturado em um fluxo modular progressivo de 5 etapas:

1. **`01_geracao_datasets.py`**: Geração e consolidação da base de dados tabular e geoespacial dos municípios paulistas.
2. **`02_esda_moran_lisa.py`**: Análise Exploratória de Dados Espaciais (ESDA), cálculo da matriz de ponderação espacial (K-NN/Queen) e índices de Moran Global e Local (LISA - *Hotspots* e *Coldspots*).
3. **`03_modelos_econometricos_sar_sem.py`**: Estimação de modelos econométricos espaciais globais (**SAR** - *Spatial Lag* e **SEM** - *Spatial Error*) para controle de autocorrelação espacial.
4. **`04_gwr_validacao_cruzada.py`**: Calibração de **Geographically Weighted Regression (GWR)** e Validação Cruzada Espacial em Blocos (*Spatial Block 5-Fold CV*) para avaliação robusta de out-of-sample.
5. **`05_mapas_interativos_web_folium.py`**: Construção de dashboard cartográfico interativo baseado em **Folium / Leaflet**, utilizando tiles do *Esri World Street Map* e renderização dos clusters LISA e coeficientes GWR locais.

---

## 📊 Principais Resultados Econométricos

* **Comparativo de Desempenho (Validação Cruzada Espacial):**
  * **OLS Clássico (In-Sample):** $R^2 = 0.6615$ | $\text{RMSE} = 0.0650$
  * **GWR Local (In-Sample):** $R^2 = 0.7349$ | $\text{RMSE} = 0.0575$
  * **OLS - Spatial Block 5-Fold CV (Out-of-Sample):** $R^2 = 0.6185$ | $\text{RMSE} = 0.0690$
  * **GWR - Spatial Block 5-Fold CV (Out-of-Sample):** $R^2 = 0.6068$ | $\text{RMSE} = 0.0700$

---

## 🌐 Visualização Interativa Web (`outputs/`)
O repositório inclui um dashboard interativo completo salvo em formato HTML na pasta `outputs/`:
* **Arquivo:** `outputs/mapa_interativo_water_energy_sp.html`
* **Recursos:** Camadas dinâmicas selecionáveis (Clusters LISA de Autorrelação Espacial e Parâmetros Locais GWR), *tooltips* interativos com dados municipais detalhados e mapa base de alta resolução (*Esri World Street Map*).

---

## 🚀 Como Executar o Projeto

1. Clone o repositório e certifique-se de utilizar um ambiente Python (como Anaconda/Spyder) com as bibliotecas exigidas (`geopandas`, `libpysal`, `esda`, `mgwr`, `spreg`, `folium`, `pandas`, `numpy`):
```bash
git clone [https://github.com/eniocancian/sp-water-energy-spatial-nexus.git](https://github.com/eniocancian/sp-water-energy-spatial-nexus.git)
cd sp-water-energy-spatial-nexus
```
2. Execute os scripts em ordem sequencial (de `01` a `05`) para processar os dados, rodar os modelos estatísticos e gerar as saídas cartográficas.
3. Abra o arquivo `outputs/mapa_interativo_water_energy_sp.html` em qualquer navegador web para interagir com o mapa dinâmico.