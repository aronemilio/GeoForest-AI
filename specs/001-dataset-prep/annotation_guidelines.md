# Directrices Técnicas de Anotación y Validación Cartográfica: GeoForest-AI
**Versión:** 1.0.0  
**Ámbito Geográfico:** Bosque Montano de Carpish y Valle del Huallaga (Huánuco, Perú)  
**Sistema de Referencia Espacial de Destino:** EPSG:32718 (WGS 84 / UTM Zona 18S)  
**Resolución Base:** 10 m/píxel (Sentinel-2 L2A)

---

## 1. Objetivo y Principios Fundamentales
Este documento define los criterios técnicos y cartográficos obligatorios para la fotointerpretación, delimitación vectorial y validación de eventos de pérdida de cobertura boscosa y quemas en el proyecto **GeoForest-AI**.

El objetivo es eliminar el ruido heurístico y garantizar que cada polígono de entrenamiento represente un disturbio forestal real verificado por especialistas, permitiendo que la arquitectura U-Net aprenda patrones fidedignos sin sobreestimar falsos positivos en mosaicos agroforestales.

---

## 2. Sistema de Referencia y Unidad Mínima Cartográfica (UMC)

### 2.1 Proyección Cartográfica
- **CRS Nativo de Trabajo:** `EPSG:32718` (UTM Zona 18 Sur).
- Si las geometrías se exportan en formato GeoJSON con coordenadas geográficas (`EPSG:4326`), el pipeline de ingestión realizará una reproyección automática rigurosa hacia `EPSG:32718` asegurando alineamiento exacto con la grilla Affine de Sentinel-2.

### 2.2 Unidad Mínima Cartográfica (UMC)
- **Área Mínima de Delimitación:** **500 m² (0.05 hectáreas)**.
- En resolución Sentinel-2 (10m × 10m = 100 m²/píxel), 500 m² equivale a **5 píxeles contiguos**.
- **Regla:** Polígonos aislados menores a 500 m² (ej. caída de un árbol individual, sombra de nube o ruido de borde) **NO deben ser digitalizados**. Si existen en la capa vectorial, el pipeline los descartará automáticamente durante el preprocesamiento.

---

## 3. Tipología de Eventos y Esquema de Atributos

Cada archivo vectorial (`.geojson` o `.shp`) debe incluir en su tabla de atributos los siguientes campos obligatorios:

| Campo | Tipo | Valores Permitidos | Descripción |
| :--- | :--- | :--- | :--- |
| `evento` | String | `deforestacion`, `quema_agricola`, `tala_selectiva` | Tipo de disturbio identificado. |
| `confianza_validacion` | Float | `0.0` a `1.0` | Grado de certeza del fotointérprete. Solo $\ge 0.80$ se usa en entrenamiento. |
| `agente_causante` | String | `expansion_agricola`, `ganaderia`, `vias`, `quema_pastos`, `desconocido` | Causa probable del disturbio. |
| `validador` | String | ID o iniciales del especialista | Identificador para trazabilidad de control de calidad. |
| `fecha_t0` | String | `YYYY-MM-DD` | Fecha de la imagen satelital pre-disturbio. |
| `fecha_t1` | String | `YYYY-MM-DD` | Fecha de la imagen satelital post-disturbio. |

### 3.1 Criterios de Clasificación por Evento

#### A. Deforestación Antropogénica (`deforestacion`)
- **Firma espectral:** Caída drástica del NIR (B08) y NDVI ($\Delta\text{NDVI} < -0.25$), aumento de reflectancia en el Rojo (B04) y exposición de suelo desnudo o rastrojo.
- **Validación óptica:** Contraste contra Google Maps Satellite de alta resolución para descartar caminos preexistentes o parcelas consolidadas.
- **Condición excluyente:** Solo se considera deforestación si el suelo en T0 pertenecía a la cobertura boscosa continua ($M_{bosque} == 1$).

#### B. Quema Agrícola / Incendio Forestal (`quema_agricola`)
- **Firma espectral:** Fuerte incremento en SWIR (B11) y caída abrupta en NIR (B08). Índice NBR con $\Delta\text{NBR} = \text{NBR}_{T0} - \text{NBR}_{T1} > 0.20$.
- **Cicatriz de quema:** Coloración oscura/ceniza característica visible en composición RGB (B04-B03-B02) y falso color SWIR (B11-B08-B04).

#### C. Control Fenológico / Falso Cambio (No Evento: $Y = 0$)
- **Definición:** Cambios de verdor debidos a la transición entre temporada de estiaje (Mayo - Octubre) y lluvias (Noviembre - Abril).
- **Regla:** La pérdida de follaje estacional en bosques caducifolios o matorrales montanos **NUNCA debe etiquetarse como deforestación**. En estos pares, el target debe ser enteramente cero ($Y = 0$).

---

## 4. Reglas Cartográficas de Digitalización

1. **Escala de Fotointerpretación:**
   - La digitalización sobre imágenes Sentinel-2 debe efectuarse a escalas entre **1:10,000** y **1:15,000**.
   - Para verificación de detalle en capas base de Google Maps Satellite, trabajar entre **1:2,500** y **1:5,000**.
2. **Delimitación de Bordes y Sombras:**
   - Digitalizar el límite efectivo del dosel forestal talado.
   - **Exclusión de Sombras Orográficas y de Copa:** No incluir dentro del polígono las sombras profundas arrojadas por laderas o árboles vecinos no talados.
3. **Regla de Rasterización en el Modelo (`all_touched=False`):**
   - El entrenamiento utiliza estrictamente `all_touched=False`.
   - Esto significa que solo los píxeles cuyo **centroide geométrico** quede dentro del polígono serán etiquetados con 1.
   - **Implicación para el fotointérprete:** Al digitalizar bordes sinuosos, no extender el polígono sobre píxeles no afectados; ajustar la línea al cuerpo principal de la masa deforestada.
4. **Topología Vectorial Estricta:**
   - Cero auto-intersecciones (`is_valid == True`).
   - Cero polígonos solapados para un mismo par temporal.
   - Geometrías multipartes (`MultiPolygon`) deben ser disueltas o separadas en entidades individuales (`Polygon`) para permitir el cálculo de UMC por cada parche individual.

---

## 5. Control de Calidad y Validación Cruzada con Geobosques (MINAM)

Antes de incorporar un conjunto de anotaciones al pipeline de entrenamiento:
1. **Cruce con Alertas Geobosques (PNCB):**
   - Todo polígono que coincida espacial y temporalmente con una alerta temprana de Geobosques recibe automáticamente una ponderación de confianza de base $\ge 0.90$.
   - Si un polígono manual discrepa diametralmente de Geobosques, requiere una segunda revisión visual con Google Maps Hybrid de alta resolución.
2. **Revisión de Mosaicos Agroforestales:**
   - En zonas de cultivo de café y cacao bajo sombra (ecotonos agroforestales), verificar que no se trate de una poda o cambio de sombrío temporal. Si el dosel superior se mantiene en $\ge 50\%$, clasificar como `degradacion_tala` o descartar si la afectación es transitoria.
