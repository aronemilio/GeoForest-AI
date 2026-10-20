# Portada animada — GeoForest-AI Huánuco

Composición [HyperFrames](https://hyperframes.heygen.com) que genera la portada
animada del panel (`app/assets/portada-bosque.mp4`).

- **Duración:** 8 s, en bucle infinito y **sin costura** (los fotogramas t=0 y
  t=8 son idénticos: mismo `md5`, PSNR infinito).
- **Lienzo:** 1600×360 a 30 fps, con el mismo arte SVG del encabezado del panel.
- **Movimiento:** niebla a la deriva, sol que respira, coníferas meciéndose y
  aves que cruzan el cuadro.
- **Decorativa:** no codifica ninguna alerta ni dato (igual que el SVG estático).

## Cómo regenerarla

Requiere Node.js y, una sola vez, Chrome/Chromium y FFmpeg en el `PATH`
(`npx hyperframes doctor` verifica el entorno).

```bash
cd videos/geoforest-portada
npx hyperframes check .        # lint + validación en runtime + layout
npx hyperframes render . -o renders/portada.mp4 -f 30 --skill=motion-graphics
```

Después, copia el resultado al panel:

```bash
cp renders/portada.mp4 ../../app/assets/portada-bosque.mp4
```

## Cómo se usa en el panel

`app/main.py` → `adorno_bosque_nuboso()` inserta el MP4 como `<video>`
silenciado, en bucle, sin controles y con esquinas redondeadas (vía
`st.markdown(..., unsafe_allow_html=True)`). Si el archivo no existe, cae al SVG
estático `_ADORNO_CARPISH_SVG` para no romper el panel.

> Nota: se usa un `<video>` y no `st.image` porque `st.image` re-codifica
> WebP→PNG estático (pierde la animación) y porque `st.video` siempre muestra
> controles de reproducción.
