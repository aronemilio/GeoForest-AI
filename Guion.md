# Guion: El Visor Interactivo y Detector Satelital (Frontend de GeoForest-AI)

## 1. ¿Cuál es la idea principal de nuestra pantalla?

Imagina que un fiscalizador ambiental necesita saber dónde se está talando o quemando el bosque en **Huánuco** (*Carpish* o el *Valle del Huallaga*), pero solo cuenta con una laptop normal de oficina. 

El **Frontend** de GeoForest-AI —construido con **Streamlit**— resuelve esto convirtiendo datos satelitales complejos en un panel visual muy fácil de usar. Su parte más importante es la conexión entre **el Mapa Interactivo** y **el Laboratorio de Detección en Vivo**.

---

## 2. ¿Cómo funciona paso a paso? (Guion de explicación)

### Paso 1: Explorar el territorio en el Mapa Interactivo
> *"Al abrir la aplicación, lo primero que vemos es un mapa vivo creado con **Folium** y **Leaflet.js**. En lugar de mostrar una imagen pesada y estática, el mapa carga capas satelitales ligeras (**WMS / TileLayers**) libres de nubes. Sobre ese mapa, el usuario puede filtrar con un clic las alertas oficiales del **MINAM (Geobosques)** y cruzarlas en tiempo real con los focos de calor de incendios que detecta la **NASA (FIRMS)**. Así, en segundos, sabemos qué zona requiere atención urgente."*

### Paso 2: Comparar el "Antes y Después" desde el espacio
> *"Cuando queremos auditar un punto sospechoso, entramos al Laboratorio de Inferencia. Aquí la interfaz toma dos fotografías reales del satélite **Sentinel-2** sobre la misma zona: una fecha anterior ($T_0$, cuando el bosque estaba sano) y una fecha reciente ($T_1$, después de la tala o quema). Usando la librería **Rasterio**, el sistema lee la luz infrarroja que refleja la vegetación para notar cambios que el ojo humano muchas veces no distingue."*

### Paso 3: Inteligencia Artificial rápida en cualquier computadora
> *"Lo más valioso de nuestro frontend es que no necesita servidores costosos ni tarjetas gráficas para analizar esas imágenes. Al presionar el botón de análisis, la interfaz envía ambas fotos a nuestra red neuronal **U-Net** empaquetada en formato **ONNX Runtime**. Este motor ligero procesa la imagen directamente en el procesador normal (**CPU**) en menos de un segundo, dibuja en pantalla la mancha exacta de deforestación con probabilidades calibradas y calcula automáticamente cuántas **hectáreas** se perdieron para descargar el reporte de fiscalización."*

---

## 3. Palabras clave para recordar

- **Streamlit:** Tecnología en Python que da vida a los botones, filtros y paneles de la página web.
- **Folium / Capas WMS:** Herramientas que dibujan el mapa interactivo y superponen las imágenes de los satélites sin trabar el navegador.
- **Sentinel-2 (Multiespectral):** Satélite europeo que captura tanto colores visibles como luz infrarroja para medir la salud del bosque.
- **ONNX Runtime:** El "motor comprimido" que permite que la Inteligencia Artificial (**U-Net**) funcione rápido en una computadora común (**CPU**).
