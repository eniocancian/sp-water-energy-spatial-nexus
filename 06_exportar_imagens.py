import os
import glob
import matplotlib.pyplot as plt
import geopandas as gpd

# 1. Definir caminhos base do projeto no Windows
base_dir = r"C:/Users/sueli/Downloads/Projetos/sp-water-energy-spatial-nexus"
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

print("Iniciando exportação de mapas reais para o portfólio...")

# 2. Busca automática por qualquer arquivo geográfico (.gpkg, .shp, .geojson) no projeto
search_pattern = os.path.join(base_dir, "**", "*.gpkg")
arquivos_gpkg = glob.glob(search_pattern, recursive=True)

if not arquivos_gpkg:
    # Tenta procurar por shapefiles (.shp) se não achar gpkg
    search_pattern_shp = os.path.join(base_dir, "**", "*.shp")
    arquivos_gpkg = glob.glob(search_pattern_shp, recursive=True)

gdf = None
if arquivos_gpkg:
    # Pega o primeiro arquivo encontrado nas pastas do projeto
    caminho_dados = arquivos_gpkg[0]
    print(f"Arquivo de dados localizado: {caminho_dados}")
    gdf = gpd.read_file(caminho_dados)
else:
    print("Aviso: Nenhum arquivo .gpkg ou .shp automático foi encontrado nas subpastas.")
    print("Por favor, informe o caminho exato do seu GeoDataFrame nas etapas anteriores.")

if gdf is not None:
    # --- MAPA 1: Clusters LISA ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Identifica coluna de cluster LISA automaticamente
    cluster_col = next((col for col in gdf.columns if 'lisa' in col.lower() or 'cluster' in col.lower() or 'quadrant' in col.lower()), None)
    
    if cluster_col:
        gdf.plot(column=cluster_col, ax=ax, legend=True, cmap='coolwarm', edgecolor='k', linewidth=0.2)
        ax.set_title(f"Clusters LISA - Nexo Água-Energia SP", fontsize=14, fontweight='bold')
    else:
        gdf.plot(ax=ax, color='lightgray', edgecolor='k', linewidth=0.2)
        ax.set_title("Clusters LISA - Nexo Água-Energia SP", fontsize=14, fontweight='bold')
        
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lisa_clusters_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Mapa LISA gerado e salvo em outputs/lisa_clusters_map.png")

    # --- MAPA 2: Coeficientes da GWR ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Identifica coluna de coeficiente GWR automaticamente
    coef_col = next((col for col in gdf.columns if 'gwr' in col.lower() or 'coef' in col.lower() or 'beta' in col.lower()), None)
    
    if coef_col:
        gdf.plot(column=coef_col, ax=ax, legend=True, cmap='YlOrRd', edgecolor='k', linewidth=0.2,
                 legend_kwds={'label': 'Coeficiente GWR Local', 'orientation': 'horizontal'})
        ax.set_title(f"Variação Espacial dos Coeficientes GWR", fontsize=14, fontweight='bold')
    else:
        gdf.plot(ax=ax, color='Blues', edgecolor='k', linewidth=0.2)
        ax.set_title("Coeficientes Locais GWR - Nexo Água-Energia SP", fontsize=14, fontweight='bold')
        
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gwr_coefficients_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Mapa GWR gerado e salvo em outputs/gwr_coefficients_map.png")
    
    print("Processo concluído com sucesso!")
else:
    print("Erro: Não foi possível carregar o GeoDataFrame. Certifique-se de que os dados foram gerados nas etapas 01 a 04.")