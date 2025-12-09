# --- analizar_velocidad.py ---
# TAREA 3: Analiza los datos del barrido de alfa y genera la Figura 4.
# (Versión con indentación corregida)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- PARÁMETROS FÍSICOS (para el título) ---
J = 1.0
D = 0.25 * J
Da = -0.10 * J
# --- FIN DE LAS LÍNEAS A AGREGAR ---

# --- PARÁMETROS DE ANÁLISIS ---
T_START_FIT = 30.0  # Empezar el ajuste después de que el pulso haya terminado
T_END_FIT = 150.0   # Terminar el ajuste antes de que el solitón muera o golpee el borde
DATA_DIR = "barrido_alfa"

# --- Función de ajuste lineal ---
def linear_func(x, a, b):
    return a * x + b

# --- Bucle de Análisis ---
if __name__ == "__main__":
    # 1. Encontrar todos los archivos de datos
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
        files.sort()
    except FileNotFoundError:
        print(f"Error: Directorio '{DATA_DIR}' no encontrado.")
        print("Por favor, ejecuta 'run_barrido_alfa.py' primero.")
        exit()

    if not files:
        print(f"Error: No se encontraron archivos .npz en '{DATA_DIR}'.")
        exit()

    print(f"Analizando {len(files)} archivos de datos...")
    
    alpha_values = []
    velocities = []

    # 2. Bucle sobre cada archivo de simulación
    for filename in files:
        # Extraer alpha del nombre del archivo
        try:
            alpha = float(filename.split('_')[-1].replace('.npz', ''))
            alpha_values.append(alpha)
        except ValueError:
            print(f"Adv: Omitiendo archivo con nombre extraño: {filename}")
            continue

        # Cargar datos
        data = np.load(os.path.join(DATA_DIR, filename))
        S_history_flat = data['S_history']
        time_points = data['time_points']
        N = S_history_flat.shape[0] // 3 # Inferir N
        
        # Reformatear datos (N_tiempos, N_sitios, 3)
        S_history = S_history_flat.T.reshape(-1, N, 3)
        sz_history = S_history[:, :, 2]
        
        # 3. Encontrar la trayectoria del solitón (posición del mínimo de Sz)
        positions = []
        times_with_soliton = []
        for t_idx, t in enumerate(time_points):
            core_indices = np.where(sz_history[t_idx] < 0.0)[0]
            if len(core_indices) > 0:
                positions.append(np.mean(core_indices))
                times_with_soliton.append(t)
        
        if not times_with_soliton:
            print(f"  -> Alpha={alpha:.3f}: No se detectó solitón (Sz nunca < 0).")
            velocities.append(np.nan)
            continue
            
        positions = np.array(positions)
        times_with_soliton = np.array(times_with_soliton)

        # 4. Filtrar datos para el ajuste lineal
        mask = (times_with_soliton > T_START_FIT) & (times_with_soliton < T_END_FIT)
        if np.sum(mask) < 2: # Necesitamos al menos 2 puntos para un ajuste
            print(f"  -> Alpha={alpha:.3f}: No hay suficientes datos en el rango de ajuste.")
            velocities.append(np.nan)
            continue

        t_fit = times_with_soliton[mask]
        pos_fit = positions[mask]
        
        # 5. Calcular la velocidad
        try:
            params, _ = curve_fit(linear_func, t_fit, pos_fit)
            velocity = params[0] # params[0] es la pendiente (velocidad)
            velocities.append(velocity)
            print(f"  -> Alpha={alpha:.3f}, Velocidad calculada = {velocity:.4f}")
        except RuntimeError:
            print(f"  -> Alpha={alpha:.3f}: Falla en el ajuste lineal.")
            velocities.append(np.nan)

    # --- 6. Graficar la Figura 4 Final (Movilidad vs. Alpha) ---
    
    # Ordenar los datos para el gráfico
    sorted_data = sorted(zip(alpha_values, velocities))
    alpha_plot = [x[0] for x in sorted_data]
    vel_plot = [x[1] for x in sorted_data]

    plt.figure(figsize=(10, 7))
    plt.plot(alpha_plot, vel_plot, 'o-', linewidth=2, markersize=8)
    
    # ¡La línea que muestra el cambio de signo!
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='v = 0')
    
    # --- BLOQUE CORREGIDO (SIN INDENTACIÓN) ---
    plt.xlabel(r'Gilbert Damping Parameter ($\alpha$)')
    plt.ylabel(r'Soliton Mobility ($\mu = dv/dh_z$)')
    plt.title(f'Mobility vs. Damping (D/J={D/J:.2f}, Da/J={Da/J:.2f})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    # --- FIN DEL BLOQUE CORREGIDO ---
    
    # --- 7. GUARDAR DATOS PARA FIGURA COMPUESTA ---
    output_data_file = "v_int_vs_alpha_data.npz"
    np.savez_compressed(output_data_file, alpha=alpha_plot, v_int=vel_plot)
    print(f"\nDatos de velocidad intrínseca guardados en {output_data_file}")
    
    plt.tight_layout()
    plt.savefig("figure_mobility_vs_alpha.png", dpi=300)
    print("\nGráfico de Movilidad vs. Alpha guardado en 'figure_mobility_vs_alpha.png'")
    plt.show()
