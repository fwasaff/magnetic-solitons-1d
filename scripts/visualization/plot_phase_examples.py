# --- crear_figura_ejemplos_fase_PRO_v2.py ---
# TAREA: Generar la figura de 3 paneles (H, SL, FM)
# (Versión B&W-safe: leyenda unificada y estilos de línea)

import numpy as np
import matplotlib.pyplot as plt
import llg_core # ¡Importamos nuestro "cerebro"!

# --- 1. Parámetros del Sistema ---
N = 200
J = 1.0

# Parámetros para cada fase (basados en nuestra Fig. 1)
params_H = {'D': 0.8 * J, 'Da': -0.1 * J, 'label': 'Helicoidal (H)'}
params_SL = {'D': 0.25 * J, 'Da': -0.10 * J, 'label': 'Soliton Lattice (SL)'}
params_FM = {'D': 0.1 * J, 'Da': -0.4 * J, 'label': 'Ferromagnetic (FM)'}

phases_to_plot = [params_H, params_SL, params_FM]

print("Calculando los 3 estados fundamentales de ejemplo...")

# --- 2. Preparación de la Figura (3 Paneles) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#fig.suptitle('Representative Ground State Configurations', fontsize=16, y=1.02)

panel_labels = ['(a)', '(b)', '(c)']
sites = np.arange(N)

for i, params in enumerate(phases_to_plot):
    ax = axes[i]
    D_val = params['D']
    Da_val = params['Da']
    
    # --- 3. Calcular el Estado Fundamental ---
    S_ground = llg_core.find_ground_state(N, J, D_val, Da_val)
    
    # --- 4. Graficar S_x, S_y, S_z (¡VERSIÓN B&W MEJORADA!) ---
    
    # S_x: Sólida, Color 0
    ax.plot(sites, S_ground[:, 0], label='$S_x$', color='C0',
            linestyle='-', linewidth=2.5, marker=None)
    
    # S_y: Discontinua, Color 1
    ax.plot(sites, S_ground[:, 1], label='$S_y$', color='C1',
            linestyle='--', linewidth=2.5, marker=None)
    
    # S_z: Punteada, Color 2
    ax.plot(sites, S_ground[:, 2], label='$S_z$', color='C2',
            linestyle=':', linewidth=2.5, marker=None)
    
    ax.set_ylabel('Spin Component')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- MEJORA: LEYENDA ÚNICA ---
    # Solo ponemos la leyenda en el primer panel (i=0)
    if i == 0:
        ax.legend(loc='upper right')
    
    # Añadir el título del panel
    title = f"{panel_labels[i]} {params['label']}: D/J={D_val/J:.2f}, Da/J={Da_val/J:.2f}"
    ax.set_title(title, loc='left')

axes[-1].set_xlabel('Chain Site ($i$)')
plt.tight_layout()

# --- Nombres de archivo consistentes ---
OUTPUT_NAME = "figure_3_phase_examples_PRO_v2"
plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
plt.savefig(f"{OUTPUT_NAME}.pdf")
print("¡Éxito! Figura de ejemplos de fase (B&W-safe) guardada.")
plt.show()
