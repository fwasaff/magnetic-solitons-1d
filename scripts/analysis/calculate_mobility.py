# --- analizar_movilidad_mu.py ---
# TAREA FINAL: Carga los datos del barrido completo, calcula la
# movilidad (mu = dv/dh) para cada alpha, y grafica mu vs. alpha.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- 1. Parámetros Fijos (DEBEN COINCIDIR CON EL RUN SCRIPT) ---
DATA_DIR = "datos_barrido_mu"
alpha_values = np.linspace(0.01, 0.20, 20)
H_DC_values = np.linspace(-0.02, 0.02, 5)

# Parámetros de análisis (ajusta si es necesario)
T_START_FIT = 30.0
T_END_FIT = 150.0

# Parámetros del gráfico (para el título)
J = 1.0
D = 0.25 * J
Da = -0.10 * J

# --- 2. Funciones Auxiliares ---
def linear_func(x, a, b):
    return a * x + b

def calculate_velocity(filename):
    """
    Carga un archivo .npz y calcula la velocidad del solitón.
    Devuelve la velocidad (float) o np.nan si falla.
    """
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"    -> Advertencia: No se encontró el archivo {filename}")
        return np.nan

    S_history_flat = data['S_history']
    time_points = data['time_points']
    N = S_history_flat.shape[0] // 3
    
    S_history = S_history_flat.T.reshape(-1, N, 3)
    sz_history = S_history[:, :, 2]
    
    # Rastrear el solitón (núcleo Sz < 0)
    positions = []
    times_with_soliton = []
    for t_idx, t in enumerate(time_points):
        core_indices = np.where(sz_history[t_idx] < 0.0)[0]
        if len(core_indices) > 0:
            positions.append(np.mean(core_indices))
            times_with_soliton.append(t)
    
    if not times_with_soliton:
        return np.nan # No se creó el solitón

    positions = np.array(positions)
    times_with_soliton = np.array(times_with_soliton)

    # Filtrar datos para el ajuste
    mask = (times_with_soliton > T_START_FIT) & (times_with_soliton < T_END_FIT)
    if np.sum(mask) < 2:
        return np.nan # El solitón murió demasiado rápido

    t_fit = times_with_soliton[mask]
    pos_fit = positions[mask]
    
    # Calcular la velocidad (pendiente)
    try:
        params, _ = curve_fit(linear_func, t_fit, pos_fit)
        return params[0] # Velocidad
    except RuntimeError:
        return np.nan # Falla en el ajuste

# --- 3. Bucle de Análisis Principal ---
if __name__ == "__main__":
    print("--- INICIANDO ANÁLISIS DE MOVILIDAD (mu) ---")
    
    alpha_plot = []
    mu_plot = []

    for alpha in alpha_values:
        print(f"Procesando alpha = {alpha:.3f} ...")
        
        velocities_for_this_alpha = []
        hdc_for_this_alpha = []
        
        for h_dc in H_DC_values:
            # Reconstruir el nombre del archivo
            alpha_str = f"{alpha:.3f}".replace('.', 'p')
            hdc_str = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
            filename = os.path.join(DATA_DIR, f"datos_a{alpha_str}_h{hdc_str}.npz")
            
            # Calcular la velocidad para este punto
            v = calculate_velocity(filename)
            
            if not np.isnan(v):
                velocities_for_this_alpha.append(v)
                hdc_for_this_alpha.append(h_dc)
                print(f"  -> h_dc={h_dc:.3f}, v={v:.4f}")
            else:
                print(f"  -> h_dc={h_dc:.3f}, v=NaN (falló el cálculo)")
        
        # --- Calcular Movilidad (mu) para este alpha ---
        if len(hdc_for_this_alpha) < 2:
            print(f"  -> Alpha={alpha:.3f}: No hay suficientes datos para calcular la movilidad (mu).")
            continue
            
        try:
            # mu = dv/dh
            params, _ = curve_fit(linear_func, hdc_for_this_alpha, velocities_for_this_alpha)
            mu = params[0] # La pendiente es la movilidad
            
            alpha_plot.append(alpha)
            mu_plot.append(mu)
            print(f"  -> ¡CÁLCULO DE MOVILIDAD EXITOSO! mu = {mu:.4f}\n")
            
        except RuntimeError:
            print(f"  -> Alpha={alpha:.3f}: Falla en el ajuste lineal para mu.")

    # --- 4. Graficar la Figura Final (mu vs. alpha) ---
    print("--- ANÁLISIS COMPLETO ---")
    print("Generando gráfico final: Movilidad (mu) vs. Amortiguamiento (alpha)")

    plt.figure(figsize=(10, 7))
    plt.plot(alpha_plot, mu_plot, 'o-', linewidth=2, markersize=8, color='C0')
    
    # La línea crítica que prueba tu hallazgo
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'$\mu = 0$')
    
    plt.xlabel(r'Parámetro de Amortiguamiento ($\alpha$)')
    plt.ylabel(r'Movilidad $\mu = dv/dh_z$')
    plt.title(f'Movilidad vs. Amortiguamiento (D/J={D/J:.2f}, Da/J={Da/J:.2f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- 5. GUARDAR DATOS PARA FIGURA COMPUESTA ---
    output_data_file = "mu_vs_alpha_data.npz"
    np.savez_compressed(output_data_file, alpha=alpha_plot, mu=mu_plot)
    print(f"\nDatos de movilidad (mu) guardados en {output_data_file}")
    
    plt.tight_layout()
    plt.savefig("figure_MOBILITY_mu_vs_alpha.png", dpi=300)
    print("Gráfico final guardado en 'figure_MOBILITY_mu_vs_alpha.png'")
    plt.show()
