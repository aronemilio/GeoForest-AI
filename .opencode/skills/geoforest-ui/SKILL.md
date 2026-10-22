---
name: GeoForest UI
description: "Guía para la interfaz Streamlit de GeoForest-AI (app/): etiquetado honesto de datos simulados, imágenes reales y salidas de un modelo sin entrenar; prohibición de valores de relleno."
---

# GeoForest UI

## Cuándo usar

- Vas a modificar `app/main.py`, `app/components/**` o `app/services/**`.
- Necesitas redactar KPIs, leyendas, popups, tablas o mensajes de estado.
- Debes decidir si un dato es real, simulado o una salida no validada del modelo.

## Ownership

- `app/main.py` pertenece a Interfaz (Grupo 1); no lo modifiques sin autorización.
- Para cambios en componentes, respeta los cambios concurrentes de otros agentes.
- Antes de editar: `git status --short`, `git diff -- <archivo>` y relee el archivo.

## Reglas duras

- **Datos simulados**: toda alerta, área y confianza de demostración debe llevar `SIMULADO` o “(Simulada)” en la etiqueta.
- **Imagen real ≠ detección**: una escena Copernicus real es solo la entrada. Etiquétala “Imagen real · modelo sin entrenar” o “Inferencia no validada”.
- **Confianza**: para salidas del modelo sin entrenar usa “Probabilidad media (sin calibrar)”; nunca “Certeza IA” ni un porcentaje de confianza.
- **Sin valores de relleno**: si no hay píxeles sobre el umbral, muestra `0,00 ha` o “Sin detecciones”; nunca 1,24 ha ni 94,2 %.
- **No elimines avisos** de modo demostrativo ni advertencias de modelo sin entrenar.
- No uses “100 %” salvo que sea una medición reproducible y documentada.

## Flujo de trabajo

1. Identifica si el dato es: simulado, imagen real o salida del modelo.
2. Aplica la tabla de `references/labeling-policy.md`.
3. Verifica sintaxis e imports:
   ```bash
   source venv/bin/activate
   python -m py_compile app/main.py app/components/*.py app/services/*.py
   python -m pytest tests/ -q
   ```
4. Reporta archivos, comandos y resultados.

## Referencias

- `references/labeling-policy.md` — tabla de etiquetas correctas y prohibidas.
