import os
import geopandas as gpd
import folium

caminho_gpkg = r"C:/Users/sueli/Downloads/Projetos/sp-water-energy-spatial-nexus/data/sp_water_energy_nexus.gpkg"

print("Lendo o GeoPackage consolidado...")
gdf = gpd.read_file(caminho_gpkg)
gdf_wgs = gdf.to_crs(epsg=4326)

centro_lat = gdf_wgs.geometry.centroid.y.mean()
centro_lon = gdf_wgs.geometry.centroid.x.mean()

# Usando o Esri World Street Map (100% gratuito, estável e livre de bloqueios 403)
mapa_web = folium.Map(
    location=[centro_lat, centro_lon],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ"
)

grupo_lisa = folium.FeatureGroup(name="Clusters LISA (Autorrelação Espacial)").add_to(mapa_web)

def estilo_lisa(feature):
    cluster = feature["properties"].get("lisa_energia_cluster", "Não Significativo")
    cor_map = {
        "High-High (Hotspot)": "#d7191c",
        "Low-Low (Coldspot)": "#2b83ba",
        "High-Low (Outlier)": "#fdae61",
        "Low-High (Outlier)": "#abd9e9",
        "Não Significativo": "#e0e0e0"
    }
    return {
        "fillColor": cor_map.get(cluster, "#e0e0e0"),
        "color": "#333333",
        "weight": 0.5,
        "fillOpacity": 0.7
    }

folium.GeoJson(
    gdf_wgs.to_json(),
    style_function=estilo_lisa,
    tooltip=folium.GeoJsonTooltip(
        fields=["name_muni", "perdas_agua_pct", "consumo_energia_agua_kwh_m3", "lisa_energia_cluster"],
        aliases=["Município:", "Perdas de Água (%):", "Consumo (kWh/m³):", "Cluster LISA:"],
        localize=True
    )
).add_to(grupo_lisa)

folium.LayerControl(collapsed=False).add_to(mapa_web)

os.makedirs("outputs", exist_ok=True)
caminho_html = os.path.abspath(os.path.join("outputs", "mapa_interativo_water_energy_sp.html"))
mapa_web.save(caminho_html)

print(f"\n[SUCESSO] Mapa interativo gerado sem erros em:\n{caminho_html}")