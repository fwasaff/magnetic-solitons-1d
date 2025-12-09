# --- crear_figura_FINAL.py ---
# TAREA: Cargar TODOS los datos y crear la figura final de 3 PANELES
# (v_int vs alpha, mu vs alpha, y v_int vs mu)
# Incluye barras de error y la región sombreada.

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Nombres de archivos de datos ---
DATA_FILE = "full_dynamics_data.npz" # El archivo que ya generamos
D_J = 0.25
Da_J = -0.10

# --- 2. Cargar los datos ---
try:
    data = np.load(DATA_FILE)
    alpha_plot = data['alpha']
    mu = data['mu']
    delta_mu = data['delta_mu']
    v_int = data['v_int']
    delta_v_int = data['delta_v_int']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{DATA_FILE}'.")
    print("Por favor, ejecuta 'analizar_movilidad_mu.py' primero.")
    exit()

print("Datos completos (v_int, mu, y errores) cargados exitosamente.")

# --- 3. VISUALIZACIÓN PROFESIONAL (3 PANELES) ---
print("Generando figura de 3 paneles...")

# --- MEJORA: plt.subplots(3, 1) para 3 paneles ---
# Ajustamos figsize para ser más alto
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

# --- Panel (a): Velocidad Intrínseca (v_int) vs. Amortiguamiento (α) ---
ax1.errorbar(alpha_plot, v_int, yerr=delta_v_int, fmt='o-', linewidth=2, 
             markersize=8, color='C0', capsize=5, label='Intrinsic Velocity ($v_{int}$)')
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_ylabel(r'Intrinsic Velocity $v_{int}$ [sites/($J^{-1}\hbar$)]', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title(f'Soliton Dynamics vs. Damping (D/J={D_J:.2f}, Da/J={Da_J:.2f})', fontsize=14)
ax1.text(-0.1, 1.0, '(a)', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='right')
ax1.legend(loc='upper right')
# Hacemos que el eje x sea visible pero sin etiquetas (se comparten)
plt.setp(ax1.get_xticklabels(), visible=False)


# --- Panel (b): Movilidad (μ) vs. Amortiguamiento (α) ---
ax2.errorbar(alpha_plot, mu, yerr=delta_mu, fmt='o-', linewidth=2, markersize=8,
             color='C3', capsize=5, label='Mobility ($\mu$)')
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'$\mu = 0$')

# --- MEJORA: Región Sombreada ---
ax2.axvspan(0.04, 0.16, color='red', alpha=0.1, label='Negative Mobility Regime')

ax2.set_xlabel(r'Gilbert Damping Parameter ($\alpha$)', fontsize=14)
ax2.set_ylabel(r'Soliton Mobility $\mu = dv/dh_z$', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')
ax2.text(-0.1, 1.0, '(b)', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='right')
plt.setp(ax2.get_xticklabels(), visible=True) # Nos aseguramos de que se vean


# --- Panel (c): Gráfico de Correlación (v_int vs. μ) ---
# ¡El gráfico "Bonus"!
ax3.errorbar(v_int, mu, xerr=delta_v_int, yerr=delta_mu,
             fmt='o', markersize=8, color='C2', capsize=5, alpha=0.7)

# Añadimos líneas de cero para ver los cuadrantes
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.axvline(0, color='gray', linestyle='--', linewidth=1)

ax3.set_xlabel(r'Intrinsic Velocity $v_{int}$', fontsize=14)
ax3.set_ylabel(r'Soliton Mobility $\mu$', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title('Correlation between Intrinsic Velocity and Mobility', fontsize=14)
ax3.text(-0.1, 1.0, '(c)', transform=ax3.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='right')


# --- Guardar y Mostrar ---
plt.tight_layout(pad=2.0)
OUTPUT_NAME = "figure_5_FINAL_COMPOSITE_3-panel"
plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
plt.savefig(f"{OUTPUT_NAME}.pdf")
print(f"\n¡Gráfico final de 3 paneles (con región sombreada) guardado en {OUTPUT_NAME}.png/pdf!")
plt.show()