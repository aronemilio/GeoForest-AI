# Política de etiquetas — GeoForest-AI UI

## Tabla de decisión

| Contexto | Etiqueta correcta | Etiqueta prohibida |
| --- | --- | --- |
| Alertas de demostración | “Alertas Simuladas Activas” | “Alertas Territoriales Activas” |
| Área de demostración | “Área Afectada Estimada (Simulada)” | “Área Afectada Estimada” |
| Confianza de demostración | “Confianza Promedio U-Net (Simulada)” | “Confianza Promedio U-Net” |
| Tabla de alertas demo | “Confianza simulada (%)” | “Confianza (%)” |
| Escena Copernicus real | “Imagen Sentinel-2 real · inferencia no validada” | “Detección Satelital Real” |
| Badge de escena real | “Imagen real · modelo sin entrenar” | “Copernicus Real (100 %)” |
| Salida del modelo | “Probabilidad media (sin calibrar)” | “Confianza 94,2 %” |
| Nivel cualitativo | “Nivel de confianza (heurístico)” | “Nivel de severidad” |
| Área sin píxeles sobre umbral | “0,00 ha” / “Sin detecciones” | “1,24 ha” |
| Expediente | “Informe demostrativo — no constituye evidencia jurídica” | “Informe pericial válido” |

## Dónde aplicar

- `app/main.py`: banner lateral, KPIs y encabezado.
- `app/components/tab_sig.py`: KPIs, leyenda, tabla y verificación del pipeline.
- `app/components/tab_inferencia.py`: tarjetas de escena, KPIs, heatmap, máscara y tabla de alertas.
- `app/components/tab_pitch.py`: tablas del simulador temporal.
- `app/components/map_view.py`: popups y tooltips.
- `app/services/expediente.py`: título, resumen y detalle por evento.

## Señales de alarma en una revisión

- Se eliminó la palabra `SIMULADO` de una etiqueta.
- Aparece “detección real”, “validado” o “100 %” sin medición.
- Hay un `else` que asigna área/confianza cuando el modelo no existe o no detecta.
- Se muestra una imagen real como si el modelo la hubiera detectado.
- Se llama “Certeza IA” a la media de una sigmoide.
