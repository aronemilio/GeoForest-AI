"""Estilos y elementos visuales de GeoForest-AI Huánuco."""

from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

_PORTADA_ANIMADA = Path(__file__).resolve().parent.parent / "assets" / "portada-bosque.mp4"

_ADORNO_CARPISH_SVG = """
<svg viewBox="0 0 1200 260" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Cordilleras del bosque nuboso de Carpish entre la niebla">
  <defs>
    <clipPath id="gfMarco">
      <rect x="1" y="1" width="1198" height="258" rx="24"/>
    </clipPath>
    <linearGradient id="gfCielo" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#EAF1E6"/>
      <stop offset="100%" stop-color="#D7E4D9"/>
    </linearGradient>
    <radialGradient id="gfSol" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#EBC57F" stop-opacity="0.95"/>
      <stop offset="55%" stop-color="#EBC57F" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#EBC57F" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <g clip-path="url(#gfMarco)">
    <rect width="1200" height="260" fill="url(#gfCielo)"/>
    <circle cx="838" cy="74" r="84" fill="url(#gfSol)"/>
    <path d="M0 150 C90 108 175 128 268 112 C372 94 438 132 540 120 C648 107 706 78 806 100 C892 119 962 138 1052 124 C1122 113 1162 130 1200 118 L1200 260 L0 260 Z" fill="#B7CCBC"/>
    <rect x="0" y="146" width="1200" height="34" fill="#F3F7F2" opacity="0.65"/>
    <path d="M0 186 C118 148 220 184 342 170 C470 155 560 192 690 178 C820 164 902 196 1012 180 C1102 167 1152 186 1200 176 L1200 260 L0 260 Z" fill="#7FA98D"/>
    <rect x="0" y="184" width="1200" height="28" fill="#EFF5EF" opacity="0.55"/>
    <path d="M0 218 C150 194 300 222 460 212 C620 202 720 228 880 216 C1010 206 1120 226 1200 216 L1200 260 L0 260 Z" fill="#2E6B4E"/>
    <g fill="#1C4A34">
      <path transform="translate(150,224)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(198,220) scale(0.8)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(240,226) scale(0.62)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(300,222) scale(0.72)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(930,222) scale(0.95)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(978,226) scale(0.7)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
      <path transform="translate(1022,220) scale(0.8)" d="M0 -48 L16 -12 L7 -12 L20 18 L-20 18 L-7 -12 L-16 -12 Z"/>
    </g>
  </g>
</svg>
"""

_DIVISOR_BOSQUE_SVG = """
<svg viewBox="0 0 1200 20" xmlns="http://www.w3.org/2000/svg">
  <path d="M0 12 C120 4 240 18 360 11 C480 4 600 17 720 10 C840 4 960 17 1080 10 C1140 7 1170 12 1200 9"
        fill="none" stroke="#BCD0BE" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

_PANTALLA_BIENVENIDA_HTML = """
<style>
  .gf-splash {
    position: fixed;
    inset: 0;
    z-index: 2147483000;
    display: grid;
    place-items: center;
    overflow: hidden;
    pointer-events: none;
    /* Fondo estático (no se anima: evita repaints). */
    background: radial-gradient(120% 95% at 50% 45%, #1B5E43 0%, #12301F 58%, #0A1E14 100%);
    animation: gfSplashFondo 2.4s cubic-bezier(0.5, 0, 0.2, 1) forwards;
    will-change: opacity;
    contain: paint;
  }
  .gf-splash__centro {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: clamp(14px, 2vw, 28px);
    /* Promoción a capa de GPU para que el scale sea compositor-only. */
    transform: translateZ(0);
  }
  /* Halo como gradiente radial (sin border/box-shadow animados: son caros). */
  .gf-splash__halo {
    position: absolute;
    top: 50%;
    left: 50%;
    width: clamp(200px, 34vw, 380px);
    aspect-ratio: 1;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(235, 197, 127, 0.50) 0%, rgba(235, 197, 127, 0.14) 42%, transparent 74%);
    transform: translate(-50%, -50%) translateZ(0);
    animation: gfSplashHalo 2.4s ease-out forwards;
    will-change: transform, opacity;
  }
  .gf-splash__letra,
  .gf-splash__texto {
    transform: translateZ(0);
  }
  /* El resplandor es estático (no se anima el text-shadow: evita repaints). */
  .gf-splash__letra {
    font-family: "Fraunces", Georgia, "Times New Roman", serif;
    font-weight: 600;
    font-size: clamp(104px, 24vw, 280px);
    line-height: 1;
    color: #EEF3EA;
    text-shadow: 0 0 30px rgba(235, 197, 127, 0.75), 0 0 72px rgba(127, 169, 141, 0.45);
    animation: gfSplashLetra 2.4s cubic-bezier(0.4, 0, 0.18, 1) forwards;
    will-change: transform, opacity;
  }
  .gf-splash__texto {
    font-family: "Karla", ui-sans-serif, system-ui, sans-serif;
    font-size: clamp(10px, 1vw, 13px);
    font-weight: 600;
    letter-spacing: 0.34em;
    text-transform: uppercase;
    color: #C7D9C5;
    animation: gfSplashTexto 2.4s ease-out forwards;
    will-change: opacity;
  }
  /* Solo se animan opacity y transform (propiedades compositor-friendly). */
  @keyframes gfSplashFondo {
    0%, 58%   { opacity: 1; }
    84%       { opacity: 0.22; }
    100%      { opacity: 0; visibility: hidden; }
  }
  @keyframes gfSplashLetra {
    0%   { opacity: 0; transform: scale(0.86); }
    16%  { opacity: 1; transform: scale(1); }
    40%  { opacity: 1; transform: scale(1.03); }
    58%  { opacity: 1; transform: scale(1.08); }
    100% { opacity: 0; transform: scale(6.5); }
  }
  @keyframes gfSplashHalo {
    0%, 34% { opacity: 0; transform: translate(-50%, -50%) scale(0.55); }
    54%     { opacity: 0.75; transform: translate(-50%, -50%) scale(1); }
    100%    { opacity: 0; transform: translate(-50%, -50%) scale(3.4); }
  }
  @keyframes gfSplashTexto {
    0%, 14%   { opacity: 0; }
    38%       { opacity: 0.9; }
    58%       { opacity: 0.9; }
    80%, 100% { opacity: 0; }
  }
  @media (prefers-reduced-motion: reduce) {
    .gf-splash { display: none; }
  }
</style>
<div class="gf-splash" aria-hidden="true">
  <div class="gf-splash__halo"></div>
  <div class="gf-splash__centro">
    <div class="gf-splash__letra">G</div>
    <div class="gf-splash__texto">GeoForest-AI Huánuco</div>
  </div>
</div>
"""

_ESTILO_TACTICO_CSS = """
<style>
  .gf-hud-card {
    background: linear-gradient(135deg, rgba(16, 32, 25, 0.90) 0%, rgba(9, 18, 14, 0.98) 100%);
    border: 1px solid rgba(46, 125, 80, 0.55);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45), 0 0 16px rgba(16, 185, 129, 0.15);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  .gf-hud-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.04em;
    color: #e6f4ea;
  }
  .gf-hud-pulse {
    width: 12px;
    height: 12px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 12px #10b981;
    animation: gfPulseDot 2s infinite ease-in-out;
  }
  @keyframes gfPulseDot {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 8px #10b981; }
    50% { transform: scale(1.35); opacity: 0.75; box-shadow: 0 0 18px #10b981; }
  }
  .gf-hud-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .gf-hud-chip {
    background: rgba(22, 45, 34, 0.85);
    border: 1px solid rgba(74, 150, 107, 0.45);
    color: #a7f3d0;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-weight: 600;
  }
  /* Animación de entrada fluida y suave sin saltos ni bloqueo de interfaz */
  @keyframes gfFluidReveal {
    0% {
      opacity: 0;
      transform: translateY(8px);
    }
    100% {
      opacity: 1;
      transform: translateY(0);
    }
  }
  .main .block-container {
    animation: gfFluidReveal 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
  }
</style>
"""


def adorno_bosque_nuboso() -> None:
    """Adorno del encabezado: cordilleras de Carpish entre la niebla, en movimiento."""
    if not _PORTADA_ANIMADA.is_file():
        st.image(_ADORNO_CARPISH_SVG, width="stretch")
        return

    video_b64 = base64.b64encode(_PORTADA_ANIMADA.read_bytes()).decode("ascii")
    st.markdown(
        f'<video src="data:video/mp4;base64,{video_b64}" autoplay muted loop '
        f'playsinline preload="auto" '
        f'style="width:100%;height:auto;display:block;border-radius:24px"></video>',
        unsafe_allow_html=True,
    )


def divisor_bosque() -> None:
    """Divisor orgánico: una línea de cresta en lugar del divisor recto por defecto."""
    st.image(_DIVISOR_BOSQUE_SVG, width="stretch")


def pantalla_bienvenida() -> None:
    """Muestra la pantalla de bienvenida (splash) al cargar la página.

    Es puramente decorativa: la letra inicial 'G' se ilumina y se expande para
    dar paso al panel. No bloquea la interacción (pointer-events: none) y se
    omite si el sistema pide movimiento reducido. Se anima solo con opacity y
    transform (compositor-friendly) para mantener fluidez.
    """
    st.markdown(_PANTALLA_BIENVENIDA_HTML, unsafe_allow_html=True)


def aplicar_estilos_tacticos() -> None:
    """Inyecta los estilos CSS tácticos y de HUD en la página."""
    st.markdown(_ESTILO_TACTICO_CSS, unsafe_allow_html=True)
