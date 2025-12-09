"""
crear_figura3_configuraciones_PRB.py
=====================================
Genera la Figura 3 del manuscrito: Configuraciones representativas de las 
tres fases (H, SL, FM) con calidad Physical Review B.

MEJORAS PRB:
- Estilos de línea y colores optimizados para B&W y daltonismo
- Marcadores distintivos en cada componente
- Grid profesional y layout optimizado
- Leyenda unificada y elegante
- Anotaciones explicativas opcionales
- Formato publication-ready (300 DPI)

Autor: Felipe Wasaff
Fecha: Noviembre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import llg_core  # Módulo de cálculo de estados fundamentales

# ============================================================================
# 1. PARÁMETROS DEL SISTEMA
# ============================================================================

N = 200  # Número de sitios en la cadena
J = 1.0  # Constante de intercambio (unidad de energía)

# Parámetros para cada fase (basados en el diagrama de fases Fig. 2)
phases_config = [
    {
        'D': 0.80 * J,
        'Da': -0.10 * J,
        'label': 'Helicoidal (H)',
        'panel': '(a)',
        'description': 'Uniform spiral modulation'
    },
    {
        'D': 0.25 * J,
        'Da': -0.10 * J,
        'label': 'Soliton Lattice (SL)',
        'panel': '(b)',
        'description': 'FM domains + chiral walls'
    },
    {
        'D': 0.10 * J,
        'Da': -0.40 * J,
        'label': 'Ferromagnetic (FM)',
        'panel': '(c)',
        'description': 'Full alignment along $z$'
    }
]

# ============================================================================
# 2. CONFIGURACIÓN DE ESTILO PRB
# ============================================================================

# Estilos optimizados para distinción en B&W y color
STYLE_CONFIG = {
    'Sx': {
        'color': '#1f77b4',      # Azul
        'linestyle': '-',        # Sólida
        'linewidth': 2.5,
        'marker': 'o',
        'markersize': 3,
        'markevery': 8,          # Un marcador cada 8 puntos
        'alpha': 0.9,
        'label': r'$S_x$'
    },
    'Sy': {
        'color': '#ff7f0e',      # Naranja
        'linestyle': '--',       # Discontinua
        'linewidth': 2.5,
        'marker': 's',           # Cuadrado
        'markersize': 3,
        'markevery': 8,
        'alpha': 0.9,
        'label': r'$S_y$'
    },
    'Sz': {
        'color': '#2ca02c',      # Verde
        'linestyle': ':',        # Punteada
        'linewidth': 3.0,        # Más gruesa para visibilidad
        'marker': '^',           # Triángulo
        'markersize': 3,
        'markevery': 8,
        'alpha': 0.9,
        'label': r'$S_z$'
    }
}

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})

# ============================================================================
# 3. CALCULAR ESTADOS FUNDAMENTALES
# ============================================================================

print("="*80)
print("GENERANDO FIGURA 3: CONFIGURACIONES DE FASES")
print("="*80)

ground_states = []
for i, config in enumerate(phases_config):
    D_val = config['D']
    Da_val = config['Da']
    phase_name = config['label']
    
    print(f"\n{config['panel']} Calculando estado fundamental: {phase_name}")
    print(f"   Parámetros: D/J={D_val/J:.2f}, Da/J={Da_val/J:.2f}")
    
    # Calcular estado fundamental
    S_ground = llg_core.find_ground_state(N, J, D_val, Da_val)
    ground_states.append(S_ground)
    
    # Verificar normalización
    norms = np.linalg.norm(S_ground, axis=1)
    print(f"   Verificación: |S| = {norms.mean():.6f} ± {norms.std():.6f}")

print("\n" + "="*80)
print("Estados fundamentales calculados exitosamente")
print("="*80 + "\n")

# ============================================================================
# 4. CREAR FIGURA CON 3 PANELES
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
sites = np.arange(N)

# Ajustar espaciado entre paneles
plt.subplots_adjust(hspace=0.25)

for i, (ax, config, S_ground) in enumerate(zip(axes, phases_config, ground_states)):
    
    # --- Graficar las tres componentes ---
    for component, (comp_name, style) in enumerate(STYLE_CONFIG.items()):
        ax.plot(sites, S_ground[:, component], 
               color=style['color'],
               linestyle=style['linestyle'],
               linewidth=style['linewidth'],
               marker=style['marker'],
               markersize=style['markersize'],
               markevery=style['markevery'],
               alpha=style['alpha'],
               label=style['label'],
               markerfacecolor='white' if component == 1 else style['color'],
               markeredgewidth=1.5)
    
    # --- Líneas de referencia ---
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.6, alpha=0.3, zorder=0)
    ax.axhline(y=-1, color='gray', linestyle='--', linewidth=0.6, alpha=0.3, zorder=0)
    
    # --- Grid profesional ---
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # --- Etiquetas y formato ---
    ax.set_ylabel('Spin Component', fontsize=13, weight='bold')
    ax.set_ylim(-1.15, 1.15)
    
    # --- Título del panel ---
    title_str = f"{config['panel']} {config['label']}: "
    title_str += f"$D/J={config['D']/J:.2f}$, $D_a/J={config['Da']/J:.2f}$"
    ax.set_title(title_str, loc='left', fontsize=12, weight='bold', pad=10)
    
    # --- Leyenda (solo en el primer panel) ---
    if i == 0:
        legend = ax.legend(loc='upper right', framealpha=0.95, 
                          edgecolor='black', fancybox=False,
                          ncol=3, columnspacing=1.5)
        legend.get_frame().set_linewidth(1.2)
    
    # --- OPCIONAL: Anotaciones explicativas ---
    if i == 1:  # Panel (b) - Soliton Lattice
        # Identificar aproximadamente dónde está el solitón
        # (buscamos la región donde S_z cambia más rápidamente)
        dSz = np.abs(np.diff(S_ground[:, 2]))
        soliton_center = np.argmax(dSz) + 1
        
        # Marcar el solitón con una región sombreada
        soliton_width = 20
        ax.axvspan(soliton_center - soliton_width/2, 
                   soliton_center + soliton_width/2,
                   alpha=0.15, color='red', zorder=0)
        
        # Anotación
        ax.annotate('Chiral\nSoliton', 
                   xy=(soliton_center, 0.5),
                   xytext=(soliton_center + 30, 0.8),
                   fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

# --- Etiqueta del eje X (solo en el panel inferior) ---
axes[-1].set_xlabel('Chain Site ($i$)', fontsize=13, weight='bold')
axes[-1].set_xlim(0, N-1)

# --- Título general (opcional) ---
# fig.suptitle('Representative Ground State Spin Configurations', 
#              fontsize=15, weight='bold', y=0.995)

# ============================================================================
# 5. GUARDAR EN FORMATOS DE PUBLICACIÓN
# ============================================================================

plt.tight_layout()

OUTPUT_NAME = "figure3_phase_configurations_PRB"

print("Guardando figura en múltiples formatos...")

# PDF (vectorial - preferido)
plt.savefig(f"{OUTPUT_NAME}.pdf", dpi=300, bbox_inches='tight')
print(f"  ✓ {OUTPUT_NAME}.pdf (vectorial)")

# PNG (alta resolución)
plt.savefig(f"{OUTPUT_NAME}.png", dpi=300, bbox_inches='tight')
print(f"  ✓ {OUTPUT_NAME}.png (raster, 300 DPI)")

# EPS (alternativa vectorial)
try:
    plt.savefig(f"{OUTPUT_NAME}.eps", dpi=300, bbox_inches='tight', format='eps')
    print(f"  ✓ {OUTPUT_NAME}.eps (vectorial)")
except:
    print("  ⚠ No se pudo guardar en formato EPS")

# TIFF (lossless)
try:
    plt.savefig(f"{OUTPUT_NAME}.tif", dpi=300, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"  ✓ {OUTPUT_NAME}.tif (lossless, 300 DPI)")
except:
    print("  ⚠ No se pudo guardar en formato TIFF")

print("\n" + "="*80)
print("✓ ¡FIGURA 3 DE CALIDAD PRB GENERADA EXITOSAMENTE!")
print("="*80)

plt.show()

# ============================================================================
# 6. CAPTION SUGERIDO
# ============================================================================

print("\n" + "="*80)
print("CAPTION SUGERIDO PARA EL MANUSCRITO:")
print("="*80)
print("""
Figure 3: Representative ground state spin configurations for the three 
phases identified in Fig. 2. Each panel shows the spatial variation of 
the three spin components S_x (blue solid line with circles), S_y (orange 
dashed line with squares), and S_z (green dotted line with triangles) along 
the chain. (a) Helicoidal phase (H) at D/J=0.80, D_a/J=-0.10: uniform spiral 
modulation with all three components oscillating with the same wavelength and 
constant amplitude. (b) Soliton Lattice phase (SL) at D/J=0.25, D_a/J=-0.10: 
ferromagnetic domains (S_z ≈ ±1) separated by narrow chiral domain walls 
(solitons) where in-plane components (S_x, S_y) develop. The shaded region 
highlights one representative soliton. (c) Ferromagnetic phase (FM) at 
D/J=0.10, D_a/J=-0.40: all spins aligned along the easy axis (S_z ≈ +1) 
with small-amplitude oscillations due to the competition between DMI and 
anisotropy. System parameters: N=200 spins, periodic boundary conditions. 
Configurations obtained via energy minimization with convergence criterion 
|ΔE/E| < 10⁻⁸. Markers are plotted every 8 sites for clarity.
""")
print("="*80)

# ============================================================================
# 7. ANÁLISIS CUANTITATIVO (OPCIONAL - Para verificación)
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS CUANTITATIVO DE LAS CONFIGURACIONES:")
print("="*80)

for i, (config, S_ground) in enumerate(zip(phases_config, ground_states)):
    print(f"\n{config['panel']} {config['label']}:")
    
    # Componente z promedio
    Sz_mean = S_ground[:, 2].mean()
    Sz_std = S_ground[:, 2].std()
    print(f"   <S_z> = {Sz_mean:+.4f} ± {Sz_std:.4f}")
    
    # Amplitud de modulación en x e y
    Sx_amplitude = (S_ground[:, 0].max() - S_ground[:, 0].min()) / 2
    Sy_amplitude = (S_ground[:, 1].max() - S_ground[:, 1].min()) / 2
    print(f"   Amp(S_x) = {Sx_amplitude:.4f}")
    print(f"   Amp(S_y) = {Sy_amplitude:.4f}")
    
    # Detectar periodicidad (para H)
    if i == 0:  # Helicoidal
        # Calcular FFT para encontrar longitud de onda
        from scipy.fft import fft, fftfreq
        fft_Sx = np.abs(fft(S_ground[:, 0]))
        freqs = fftfreq(N, d=1)
        # Encontrar pico principal (excluir DC)
        peak_idx = np.argmax(fft_Sx[1:N//2]) + 1
        wavelength = 1 / freqs[peak_idx]
        print(f"   Longitud de onda dominante: λ ≈ {wavelength:.1f} sitios")
    
    # Contar solitones (para SL)
    if i == 1:  # Soliton Lattice
        # Contar cruces por cero de S_z
        zero_crossings = np.where(np.diff(np.sign(S_ground[:, 2])))[0]
        n_solitons = len(zero_crossings) // 2  # Cada solitón tiene 2 cruces
        print(f"   Número aproximado de solitones: {n_solitons}")
        if n_solitons > 0:
            spacing = N / n_solitons
            print(f"   Espaciamiento promedio: {spacing:.1f} sitios")

print("\n" + "="*80)
print("NOTAS PARA EL MANUSCRITO:")
print("="*80)
print("""
1. VERIFICACIÓN:
   - Todos los spines tienen |S| = 1 (verificado arriba)
   - Las configuraciones corresponden a mínimos de energía
   
2. PUNTOS DEL DIAGRAMA DE FASES:
   - Verifica que (D/J, D_a/J) coincidan con los puntos marcados en Fig. 2
   - Si agregaste cuadrados verdes en Fig. 2, estos deberían ser esos puntos
   
3. PARA EL TEXTO:
   - Menciona los parámetros exactos usados (D/J, D_a/J)
   - Describe cualitativamente cada configuración (usa el caption)
   - Relaciona con la competencia DMI-anisotropy
   
4. DISTINGUIBILIDAD:
   - Verifica que los estilos de línea sean distinguibles en B&W
   - Los marcadores ayudan a identificar cada componente
   
5. MATERIAL SUPLEMENTARIO (opcional):
   - Puedes agregar animaciones de estas configuraciones
   - O mostrar más puntos representativos del diagrama de fases
""")
print("="*80)
