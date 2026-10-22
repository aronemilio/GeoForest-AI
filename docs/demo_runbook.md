# Guía de exposición — GeoForest-AI Huánuco

> Documento de preparación para la exposición ante jurados, SERFOR y FEMA. Esta guía detalla **el recorrido narrativo, qué destacar y cómo demostrar la plataforma**.
> **Advertencia de comunicación:** el score del modelo cuenta con calibración post-entrenamiento mediante Temperature Scaling ($T > 1.0$) para mitigar la sobreconfianza. Checkpoint actual: época 8 de 12. Validación OSCD por ciudades no vistas (usada para seleccionar la época; no es un test ciego): **F1 22.54 %, IoU 12.70 %, precisión 39.69 %, recall 15.74 %**. Calibración medida en OSCD: Brier **0.0332** (baseline 0.0348), ECE-15 **0.0251**, MCE-15 **0.4485**; `p ≥ 0.80` → **44.49 %** de precisión empírica y `p ≥ 0.90` → **50.55 %**. La métrica de Huánuco (F1 78.84 %, IoU 65.07 %) usa un pseudo-label generado con las mismas reglas del entrenamiento y **no es validación independiente**. Decir «prototipo en validación», no «detector validado». La serie de 8 semanas del Modo Pitch es **demostrativa** (bins fijos de 2026), y la columna «Score heurístico de persistencia espacial (%)» de las alertas es **heurística, no oficial ni calibrada**. No presentar cifras de 98–99 % como confianza del modelo.

## Checklist de arranque (10 min antes)

1. **Terminal — panel principal Streamlit:**
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   source venv/bin/activate
   streamlit run app/main.py --server.headless true --server.port 8501
   ```
   Verificar que `http://127.0.0.1:8501` responda con HTTP 200.

2. **Navegador:** abrir `http://127.0.0.1:8501` en modo incógnito/limpio. Dejar que concluya la pantalla de bienvenida interactiva (~3 s).

3. **Verificar que no haya procesos huérfanos:**
   ```bash
   lsof -i :8501 || ss -tulpn | grep 8501
   ```

---

## Recorrido propuesto (≈ 10–12 min)

### A. Primeras 60 s — Pantallazo visual y narrativa territorial

- La **pantalla de bienvenida (splash)** presenta el isotipo forestal animado en CSS puro. Luego se despliega la **portada animada** (`app/assets/portada-bosque.mp4`) en bucle continuo de 8 segundos.
- Explicar el ámbito territorial prioritario:
  - **Bosque Montano de Carpish (Chinchao):** Ecosistema de neblina de alta fragilidad y endemismo.
  - **Valle del Huallaga (Chinchao, Churubamba, Acomayo):** Frente activo de expansión agroforestal y quemas.
- Destacar el origen de datos: **PNCB Geobosques (MINAM)** a partir de **Sentinel-2 (10 m)** e integración en vivo de focos de calor **NASA FIRMS (sensor VIIRS)**.

### B. KPIs y Consola SIG (Mapa Multicapa)

- **Tarjetas de Indicadores:**
  - *Alertas Satelitales Activas:* 2,853 eventos de deforestación agrupados del PNCB Geobosques.
  - *Área Afectada Registrada:* Superficie calculada a partir de alertas oficiales, en hectáreas.
  - *Severidad Alta:* Eventos clasificados por tamaño y prioridad para inspección FEMA.
  - *Score de Consistencia:* valor normalizado del modelo; la app lo muestra como «Score heurístico (sin calibrar)», pero **no es una probabilidad calibrada ni una precisión validada**. Medición OSCD: `p ≥ 0.80` → 44.49 % de precisión empírica; `p ≥ 0.90` → 50.55 %.
- **Cartografía Folium Abierta (Sin API Keys bloqueantes):**
  - Mostrar la conmutación de capas: *Satélite real (Esri World Imagery)*, *Sentinel-2 Cloudless EOX*, *Copernicus CDSE* y *Focos Térmicos NASA VIIRS*.
  - Inspeccionar un polígono de alerta sobre el mapa para mostrar el popup con identificador único (e.g. `PNCB-HCO-0001`), coordenadas UTM 18S y área.

### C. Consola de Misión Satelital (Laboratorio de Inferencia U-Net)

- Abrir la pestaña **🔬 Laboratorio de Inferencia**:
  - **Selector de Misiones Satelitales Reales:** Demostrar cómo conmutar entre las 5 escenas reales de Sentinel-2 L2A (`Huánuco Completa Tile 18LUQ`, `Carpish Norte`, `Valle Huallaga Sur`, `Aguas Claras` y `Brasilia`).
  - **Ficha de Telemetría Orbital:** Explicar los metadatos de la misión (resolución de 10 m/px, bandas B04 Rojo, B08 NIR y B11 SWIR, temporalidad multitemporal $T_0$ vs $T_1$).
  - **Previsualización Dual en Vivo:** Comparar las imágenes en falso color infrarrojo pre y post-evento.
- **Ejecución en Tiempo Real (CPU):**
  - Pulsar **«⚡ Ejecutar Inferencia U-Net en Tiempo Real»**.
  - Resaltar la **latencia de CPU**: ~40 ms para una inferencia ONNX de 256×256 en este equipo; el pipeline completo puede tardar más por E/S y post-proceso.
  - Explicar la **Umbralización Adaptativa de Otsu**: el algoritmo calcula automáticamente el corte bimodal óptimo entre fondo y cambio sin depender de umbrales manuales arbitrarios.
  - Destacar el **Filtro Biofísico Multitemporal**: filtra candidatos que no cumplen con caída de dosel ($\Delta\text{NDVI} \ge 0.10$) o incremento térmico/desecación ($\Delta\text{NBR} \ge 0.15$), con el objetivo de reducir falsos positivos.
  - Mostrar la cuadrícula de 4 vistas: Falso Color Infrarrojo, Índice NDVI, Heatmap de score normalizado (no calibrado; no interpretar como precisión) y Máscara Binaria de candidatos.

### D. Modo Pitch: Línea de Tiempo Semanal

- Cambiar a la vista **🌟 Presentación Ejecutiva Unificada**.
- Recorrer el deslizador temporal **demostrativo** de 8 semanas (bins fijos de agosto a septiembre de 2026; no son observaciones semanales reales), ilustrando cómo evoluciona la tasa semanal de deforestación y la respuesta de fiscalización.

### E. Cierre y Cadena de Custodia Digital

- Descargar la capa vectorial en **GeoJSON** para interoperabilidad con QGIS o ArcGIS.
- Descargar el **Expediente Técnico Digital**: destacar el bloque criptográfico con **hash SHA-256**, que permite verificar la integridad técnica y la trazabilidad; **no certifica la inmutabilidad de la evidencia ni sustituye una pericia oficial** de FEMA/SERFOR.
- Respaldar la solidez del código: **103 pruebas unitarias automatizadas con 100% de éxito**.

---

## Puntos clave a destacar ante el jurado

- ✅ **Datos de entrada reales:** Las capas operativas usan Copernicus Sentinel-2 L2A, Geobosques MINAM y NASA FIRMS. El Modo Pitch conserva una serie demostrativa, y la columna «Score heurístico de persistencia espacial (%)» de las alertas es heurística, no oficial ni calibrada. No usar las cifras de 98–99 % como si fueran calibración del modelo.
- ✅ **Inferencia en CPU (prototipo):** El motor U-Net corre en ONNX Runtime optimizado (~40 ms por parche de 256×256 en este equipo), apto para servidores de borde o estaciones de campo sin GPU costosa.
- ✅ **Rigor Matemático:** Binarización adaptativa mediante el método de Otsu y filtrado biofísico multitemporal ($\Delta\text{NBR} / \Delta\text{NDVI}$).
- ✅ **Interoperabilidad GIS Completa:** Proyecciones UTM WGS84 EPSG:32718 y formatos estándar GeoTIFF / GeoJSON.

---

## Diagnóstico rápido si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| Panel no carga / Connection refused | Servidor apagado | Arrancarlo con `streamlit run app/main.py --server.port 8501`. |
| Puerto 8501 ocupado | Proceso previo colgado | Matar proceso con `kill -9 $(lsof -t -i:8501)`. |
| Advertencia de caché | Recarga en curso | Recargar la pestaña del navegador con `Ctrl+Shift+R`. |
| Focos NASA no cargan | Sin conexión externa | El sistema usa la capa satelital local y caché automática sin detenerse. |

## Archivos clave de consulta

- `app/main.py` y `app/components/` (interfaz, Consola de Misión y modo Pitch).
- `app/services/` (`copernicus.py`, `alerts_data.py`, `expediente.py`, `nasa_firms.py`).
- `src/infer.py` (motor ONNX, Otsu, normalización de scores y filtro biofísico).
- `src/preprocess.py` (validación de rásteres, remuestreo explícito e índices espectrales).
- `tests/` (92 pruebas unitarias automatizadas).
