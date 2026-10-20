# Guía de exposición — GeoForest-AI Huánuco

> Documento de preparación. Lee el README (sección *Limitaciones importantes*) antes de presentar. Esta guía dice **qué mostrar, en qué orden y qué NO mostrar**.

## Checklist de arranque (10 min antes)

1. **Terminal 1 — panel Streamlit:**
   ```bash
   cd /home/emi/Proyectos/GeoForest-AI-Huanuco
   source .venv/bin/activate
   python -m streamlit run app/main.py --server.headless true --server.port 8501 --server.fileWatcherType none
   ```
   Verificar `http://127.0.0.1:8501` responde.

2. **Terminal 2 — visor web React (opcional pero recomendado):**
   ```bash
   cd /home/emi/Proyectos/GeoForest-AI-Web
   npm run dev -- --port 5174 --host
   ```
   Verificar `http://127.0.0.1:5174/` responde. Si no, la pestaña «🌐 Visor Táctico Web» muestra un aviso honesto (no está roto, solo no se inició).

3. **Navegador:** abrir `http://127.0.0.1:8501` en modo limpio (sin caché problemática). Dejar que termine la animación de la pantalla de bienvenida (~3 s).

4. Verificar que no haya un servidor Streamlit anterior: `pgrep -af streamlit`.

## Recorrido propuesto (≈ 10–12 min)

### A. Primeras 60 s — pantallazo visual (tab «🗺️ Panel de Monitoreo GIS»)

- La **splash** aparece con la letra «G» que se ilumina y se expande. Luego la **portada animada** del encabezado (niebla, coníferas meciéndose, aves) en bucle sin costura.
- Presenta el tema visual «Niebla de dosel» y el sidebar de filtros (zona, tipo de evento, severidad).
- **Deja visible el aviso amarillo** «Modo Demostrativo Activo»: los datos son simulados. Dilo explícitamente.
- Muestra el mapa con los polígonos de alerta y las fichas. Los marcadores indican `origen_dato = SIMULADO`, área y confianza (simulada).

### B. KPIs (tab GIS, columna derecha)

- «Alertas Simuladas Activas», «Área Afectada Estimada (Simulada)», «Severidad Alta», «Confianza Promedio U-Net (Simulada)» — los cuatro llevan «Simulada» en la etiqueta.
- Explica que el U-Net está implementado y verificado por pruebas, pero **el modelo servido está sin entrenar** (ver C).

### C. Verificación técnica del motor (misma pestaña, sección inferior)

- **Botón «🔧 Verificar pipeline ONNX (modelo sin entrenar)»** → se ejecuta en ~0,6 s. Muestra el tensor multitemporal, el CRS, el peso real (31,1 MB) y **0,00 ha / 0,0 %**, porque el modelo es de referencia.
- **No digas que esto es una detección.** El aviso rojo lo deja claro. Mensaje preparado: «Esto verifica que la cadena recorte→tensor→ONNX→superficie funciona y cuánto tarda. El modelo de referencia se sustituye por uno entrenado cuando haya etiquetas validadas de Geobosques».
- Menciona la inferencia por ventanas deslizantes y la fusión suave (Gaussiana/lineal/constante) como capacidad del motor, y las 77 pruebas automatizadas (46 GIS, 11 U-Net, 16 inferencia, 4 simulación).

### D. Tab «⏱️ Modo Pitch (Simulador Temporal)»

- Simulación temporal por semanas, determinista (misma semilla Mulberry32 bit-a-bit con el visor React).
- Útil para contar la narrativa: acumulación de alertas, expansión semanal. Recalca que son datos simulados.

### E. Tab «🧠 Benchmark U-Net ONNX (métricas simuladas)»

- Primero lee el **aviso amarillo**: «estas métricas NO están medidas». Los números son una plantilla de referencia.
- El peso real (31,1 MB FP32) sí es medido. Todo lo demás (IoU 88,4 %, F1 93,8 %, latencia 18,4 ms) es maquetación.
- Explica el camino hacia métricas reales: etiquetas de Geobosques, entrenamiento validado por un especialista.

### F. Tab «🌐 Visor Táctico Web (React :5174)»

- Si está en marcha, muestra el iframe con el barrido de radar y la telemetría. Resalta que comparte la misma semilla determinista con el panel.
- Si no arrancó, recuerda la degradación honesta.

### G. Cierre

- Descarga de CSV/GeoJSON y el expediente demostrativo (`.txt` con hash SHA-256 de integridad).
- Hoja de ruta del proyecto (expander al final): Etapas 1–4.
- Pregunta final: «El motor y el panel están listos; faltan datos reales + entrenamiento + validación».

## Qué NO decir / NO mostrar

- ❌ No afirmes que el U-Net detecta bosque (no está entrenado).
- ❌ No leas las cifras de IoU/F1/latencia como medidas.
- ❌ No llames «cuantizado (Int8)» al modelo: es FP32, sin PTQ.
- ❌ No digas «54,3 parches/segundo» ni «cadena de custodia probatoria»: son parches inventados.
- ❌ No abras la pestaña del visor si el servidor 5174 no responde (el aviso lo maneja solo).
- ❌ No presentes el expediente FEMA/SERFOR como un informe pericial válido (son datos simulados).
- ❌ No dejes que un jurado pulse el botón «Verificar pipeline» sin contexto.
- ❌ No ocultes el «Modo Demostrativo Activo».

## Diagnóstico rápido si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| Panel no carga / 404 | Servidor no iniciado | Arrancarlo con `python -m streamlit` (no con el script de consola). |
| `ModuleNotFoundError: No module named 'src'` | Lanzado sin `python -m` | Reiniciar con el comando de la guía. |
| Pestaña radar vacía | React :5174 apagado | Levantarlo o usarla solo como aviso. |
| Animación se congela | Ya reproduciéndose una vez por carga | Es por diseño (CSS, una reproducción por carga de página). |
| Discrepancia de código con la copia commiteada | Otro agente editó después | Revisar `git status` y `git diff`. |
| Resultado del pipeline `0,00 ha` | Modelo sin entrenar (correcto) | Explicar con el mensaje preparado. |
<|end|> El panel cambió (edits ajenos) | Trabajo multiagente | Ver `AGENTS.md` y preguntar antes de sobrescribir. |

## Files clave para el jurado

- `app/main.py` (interfaz, 4 pestañas, splash y portada).
- `src/preprocess.py` (35+ pruebas GIS; este módulo está completo).
- `src/unet.py` y `src/infer.py` (arquitectura y motor; pruebas).
- `src/sim/` (simulador determinista compartido con el visor React).
- `tests/` (77 pruebas automatizadas).
- `videos/geoforest-portada/` + `app/assets/` (animaciones).
- `.streamlit/config.toml` (tema).
- `AGENTS.md` (protocolo multiagente y mapa de propiedad).

## Al cerrar la exposición

- Comandos de cierre: `Ctrl+C` en ambas terminales. No te preocupes por los cambios, están commiteados.
