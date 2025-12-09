"""
crear_figura5_dynamics_PRB.py
==============================
Genera la Figura 5 del manuscrito: Dinámica de solitones vs damping
Incluye:
- Panel (a): Velocidad intrínseca vs α
- Panel (b): Movilidad vs α con región sombreada
- Panel (c): Correlación v_int vs μ

Calidad Physical Review B
Autor: Felipe Wasaff
Fecha: Noviembre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# 1. CARGAR O GENERAR DATOS
# ============================================================================

# OPCIÓN A: Si tienes los datos guardados, cárgalos
# alpha_values = np.load('alpha_values.npy')
# v_int_values = np.load('v_int_values.npy')
# mobility_values = np.load('mobility_values.npy')
# v_int_errors = np.load('v_int_errors.npy')
# mobility_errors = np.load('mobility_errors.npy')

# OPCIÓN B: Datos de ejemplo (REEMPLAZA CON TUS DATOS REALES)
# Estos son valores aproximados basados en la figura que compartiste
alpha_values = np.array([
    0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20
])

v_int_values = np.array([
    -0.74, -0.41, -0.28, -0.26, -0.26, -0.26, -0.26, -0.26, -0.26,
    -0.26, -0.27, -0.28, -0.28, -0.29, -0.24, -0.29, -0.37, -0.43, -0.66
])

mobility_values = np.array([
    4.8, -0.5, -1.8, -2.0, -2.5, -2.8, -3.0, -2.0, -1.8,
    -2.2, -2.5, -3.5, -3.8, -4.0, -3.0, 3.0, -8.0, -11.0, -23.0
])

# Errores (barras de error)
v_int_errors = np.array([
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.06, 0.08, 0.10, 0.12
])

mobility_errors = np.array([
    0.3, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.5, 1.5, 4.0, 2.0, 3.0, 8.0
])

# ============================================================================
# 2. PARÁMETROS DE CONFIGURACIÓN
# ============================================================================

DPI = 300
FIGSIZE = (15, 5)  # Ancho para 3 paneles lado a lado

# Identificar región de movilidad negativa
negative_mobility_mask = (alpha_values >= 0.04) & (alpha_values <= 0.16)
alpha_negative_start = 0.04
alpha_negative_end = 0.16

# ============================================================================
# 3. CONFIGURAR ESTILO PRB
# ============================================================================

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# ============================================================================
# 4. CREAR FIGURA CON 3 PANELES
# ============================================================================

fig = plt.figure(figsize=FIGSIZE)
gs = GridSpec(1, 3, figure=fig, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])  # Panel (a): v_int vs α
ax2 = fig.add_subplot(gs[0, 1])  # Panel (b): μ vs α
ax3 = fig.add_subplot(gs[0, 2])  # Panel (c): μ vs v_int

# ============================================================================
# 5. PANEL (a): VELOCIDAD INTRÍNSECA vs DAMPING
# ============================================================================

ax1.errorbar(alpha_values, v_int_values, yerr=v_int_errors,
             fmt='o-', color='#1f77b4', linewidth=2, markersize=7,
             capsize=4, capthick=1.5, label='Intrinsic Velocity ($v_{int}$)',
             ecolor='#1f77b4', elinewidth=1.5)

# Línea de referencia en y=0
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Etiquetas y formato
ax1.set_xlabel(r'Gilbert Damping Parameter ($\alpha$)', fontsize=13, weight='bold')
ax1.set_ylabel(r'Intrinsic Velocity $v_{int}$ [sites/$J^{-1}\hbar$]', 
               fontsize=13, weight='bold')
ax1.set_title('(a)', fontsize=14, weight='bold', loc='left')
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax1.legend(loc='lower right', framealpha=0.95)

# Límites
ax1.set_xlim(0.01, 0.21)
ax1.set_ylim(-0.85, 0.05)

# ============================================================================
# 6. PANEL (b): MOVILIDAD vs DAMPING (con región sombreada)
# ============================================================================

# Región sombreada de movilidad negativa
ax2.axvspan(alpha_negative_start, alpha_negative_end, 
            alpha=0.25, color='red', zorder=0,
            label='Negative Mobility Regime')

# Línea de μ = 0
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, 
            alpha=0.8, zorder=1, label=r'$\mu = 0$')

# Datos de movilidad
ax2.errorbar(alpha_values, mobility_values, yerr=mobility_errors,
             fmt='o-', color='#d62728', linewidth=2, markersize=7,
             capsize=4, capthick=1.5, label='Mobility ($\mu$)',
             ecolor='#d62728', elinewidth=1.5, zorder=5)

# Etiquetas y formato
ax2.set_xlabel(r'Gilbert Damping Parameter ($\alpha$)', fontsize=13, weight='bold')
ax2.set_ylabel(r'Soliton Mobility $\mu = dv/dh_z$', fontsize=13, weight='bold')
ax2.set_title('(b)', fontsize=14, weight='bold', loc='left')
ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax2.legend(loc='upper right', framealpha=0.95, fontsize=9)

# Límites
ax2.set_xlim(0.01, 0.21)
ax2.set_ylim(-32, 8)

# ============================================================================
# 7. PANEL (c): CORRELACIÓN PARAMÉTRICA μ vs v_int
# ============================================================================

# Colormap según α (del azul al rojo)
colors = plt.cm.coolwarm(alpha_values / alpha_values.max())

# Scatter plot con barras de error bidimensionales
for i in range(len(alpha_values)):
    ax3.errorbar(v_int_values[i], mobility_values[i],
                xerr=v_int_errors[i], yerr=mobility_errors[i],
                fmt='o', color=colors[i], markersize=8,
                capsize=3, capthick=1.5, alpha=0.8,
                ecolor=colors[i], elinewidth=1.2)

# Líneas de referencia
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Etiquetas y formato
ax3.set_xlabel(r'Intrinsic Velocity $v_{int}$ [sites/$J^{-1}\hbar$]', 
               fontsize=13, weight='bold')
ax3.set_ylabel(r'Soliton Mobility $\mu$', fontsize=13, weight='bold')
ax3.set_title('(c)', fontsize=14, weight='bold', loc='left')
ax3.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

# Colorbar para mostrar α
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=alpha_values.min(), 
                                             vmax=alpha_values.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, pad=0.02, fraction=0.046)
cbar.set_label(r'$\alpha$', rotation=0, labelpad=15, fontsize=12, weight='bold')

# Límites
ax3.set_xlim(-0.8, 0.0)
ax3.set_ylim(-30, 8)

# ============================================================================
# 8. TÍTULO GENERAL (OPCIONAL)
# ============================================================================

#fig.suptitle('Soliton Dynamics vs. Damping (D/J=0.25, Da/J=-0.10)', 
#             fontsize=15, weight='bold', y=1.02)

# ============================================================================
# 9. GUARDAR FIGURA
# ============================================================================

plt.tight_layout()

OUTPUT_NAME = "figure5_soliton_dynamics_PRB"

print("="*80)
print("GENERANDO FIGURA 5: DINÁMICA DE SOLITONES")
print("="*80)

# PDF
plt.savefig(f"{OUTPUT_NAME}.pdf", dpi=DPI, bbox_inches='tight')
print(f"✓ Guardado: {OUTPUT_NAME}.pdf (vectorial)")

# PNG
plt.savefig(f"{OUTPUT_NAME}.png", dpi=DPI, bbox_inches='tight')
print(f"✓ Guardado: {OUTPUT_NAME}.png (raster, {DPI} DPI)")

# EPS
try:
    plt.savefig(f"{OUTPUT_NAME}.eps", dpi=DPI, bbox_inches='tight', format='eps')
    print(f"✓ Guardado: {OUTPUT_NAME}.eps (vectorial)")
except:
    print("⚠ No se pudo guardar en formato EPS")

print("\n" + "="*80)
print("CAPTION SUGERIDO:")
print("="*80)
print("""
Figure 5: Soliton dynamic properties as a function of Gilbert damping α 
for D/J=0.25, D_a/J=-0.10. (a) Intrinsic velocity v_int (velocity at h_z=0) 
shows a non-monotonic dependence on α, with an abrupt transition near α≈0.04 
and increased fluctuations for α>0.16. (b) Mobility μ=dv/dh_z reveals a 
complex relationship with α, including a robust sign change near α≈0.04 
(first zero crossing) and large fluctuations for α>0.16. The red shaded 
region marks the negative mobility regime (0.04<α<0.16). Error bars represent 
standard deviation from N=5 independent simulations. (c) Parametric plot of 
mobility vs. intrinsic velocity with α varying implicitly (color scale). The 
multi-valued, non-monotonic correlation demonstrates the failure of rigid-
particle models and confirms regime-dependent internal deformations. The 
discrete clustering of data points reveals distinct dynamical regimes 
corresponding to different soliton structures.
""")
print("="*80)

plt.show()

# ============================================================================
# 10. INSTRUCCIONES DE USO
# ============================================================================

print("\n" + "="*80)
print("INSTRUCCIONES:")
print("="*80)
print("""
1. REEMPLAZAR DATOS:
   - Este código usa datos de ejemplo (líneas 23-49)
   - Reemplázalos con tus datos reales desde archivos .npy o arrays

2. FORMATO DE DATOS REQUERIDO:
   - alpha_values: array 1D con valores de α
   - v_int_values: array 1D con velocidades intrínsecas
   - mobility_values: array 1D con movilidades
   - v_int_errors: array 1D con errores en v_int
   - mobility_errors: array 1D con errores en μ

3. AJUSTAR PARÁMETROS:
   - Líneas 53-54: Región de movilidad negativa (0.04 a 0.16)
   - Ajusta según tus datos reales

4. PARA CARGAR TUS DATOS:
   Descomenta líneas 18-22 y ajusta nombres de archivo

5. VERIFICAR:
   - Que los límites de los ejes sean apropiados
   - Que las unidades coincidan con tu manuscrito
   - Que los colores sean distinguibles en escala de grises
""")
print("="*80)
