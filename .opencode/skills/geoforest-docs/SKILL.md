---
name: GeoForest Docs
description: Guía para mantener README.md, AGENTS.md y docs/ de GeoForest-AI sincronizados con el estado real del repositorio (versiones, modelo, datos, pruebas y comandos).
---

# GeoForest Docs

## Cuándo usar

- Vas a editar `README.md`, `AGENTS.md` o `docs/**`.
- Detectas una cifra que no cuadra (tamaño del modelo, canales, versiones, número de pruebas).
- Necesitas reconciliar el inventario de superficie de ataque o el estado real de los artefactos.

## Reglas duras

- **No copies cifras de documentación antigua**: verifícalas con comandos antes de escribirlas.
- Distingue siempre **simulado** de **real** y **modelo sin entrenar** de **modelo validado**.
- El entorno del proyecto es `venv` (no `.venv`); Python 3.10.20.
- No cambies `requirements.txt` sin autorización: es un archivo de coordinación.
- Si `requirements.txt` no coincide con el entorno, repórtalo; no lo “arregles” silenciosamente.
- No presentes `src/train.py` como existente si no está en el repositorio.

## Flujo de trabajo

1. Verifica el estado real:
   ```bash
   git status --short
   source venv/bin/activate
   python --version
   python -m pytest tests/ --collect-only -q | tail -1
   stat -c '%s %n' models/unet_geoforest.onnx models/unet_best.pth
   du -sh data
   ```
2. Contrasta con `references/repo-facts.md`.
3. Edita solo las afirmaciones obsoletas; conserva el resto.
4. Si cambiaste un comando, pruébalo en una shell limpia.

## Referencias

- `references/repo-facts.md` — inventario verificado y comandos de comprobación.
