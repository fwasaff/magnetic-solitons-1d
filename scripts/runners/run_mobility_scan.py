# --- run_barrido_completo.py ---
# TAREA DEFINITIVA: Generar la curva de Movilidad (mu) vs. Amortiguamiento (alpha).
# Esto implica un "barrido de barridos":
#   Bucle 1: sobre alpha
#   Bucle 2: sobre h_dc (para calcular la pendiente dv/dh)

import numpy as np
import os
from scipy.integrate import solve_ivp
import time
import llg_core  # Importamos nuestro módulo de física

# --- 1. Parámetros Fijos de la Física ---
N = 200
J = 1.0
D = 0.25 * J
Da = -0.10 * J  # Anisotropía de EJE FÁCIL
gamma = 1.0
OUTPUT_DIR = "datos_barrido_mu"

# Parámetros del Pulso (para crear el solitón)
# (Usamos el pulso en X que ya validamos)
h_pulse_params = {
    'h0': -10.0 * J, 't0': 2.0, 'tau': 0.5,
    'i0': N // 2, 'sigma': 3.0
}

# --- 2. Parámetros del BARRIDO ---
# El barrido sistemático que el réferi esperaba
alpha_values = np.linspace(0.01, 0.20, 20)  # 20 puntos para alpha
H_DC_values = np.linspace(-0.02, 0.02, 5)  # 5 puntos de campo para la pendiente

# Parámetros de Simulación
T_MAX = 200.0
DT_SAVE = 0.5  # Guardar datos con menos frecuencia

# --- 3. Funciones de Campo Externo ---
def pulsed_field_generator(t, N, h0, t0, tau, i0, sigma):
    """
    Genera el pulso de campo GAUSSIANO en X para crear el solitón.
    """
    time_profile = h0 * np.exp(-((t - t0)**2) / (2 * tau**2))
    indices = np.arange(N)
    space_profile = np.exp(-((indices - i0)**2) / (2 * sigma**2))
    
    field = np.zeros((N, 3))
    field[:, 0] = time_profile * space_profile  # Pulso en EJE X
    return field

def h_total(t, N, **kwargs):
    """
    Función de campo externo total que el solver usará.
    Combina el pulso (en X) y el campo DC (en Z).
    """
    h_pulse = pulsed_field_generator(t, N, **kwargs['pulse_params'])
    
    h_dc_vec = np.zeros((N, 3))
    h_dc_vec[:, 2] = kwargs['h_dc']  # Campo DC constante en Z
    
    return h_pulse + h_dc_vec

# --- 4. Ejecución del Barrido de Barridos ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directorio '{OUTPUT_DIR}' creado.")

    print("--- INICIANDO BARRIDO COMPLETO DE MOVILIDAD ---")
    print(f"Parámetros fijos: N={N}, D/J={D/J:.2f}, Da/J={Da/J:.2f}")
    print(f"Iterando sobre {len(alpha_values)} valores de alpha.")
    print(f"Iterando sobre {len(H_DC_values)} valores de h_dc.")
    print(f"Total de simulaciones a ejecutar: {len(alpha_values) * len(H_DC_values)}")
    print("-" * 60)

    total_sim_count = 1
    total_sims = len(alpha_values) * len(H_DC_values)

    # --- BUCLE PRINCIPAL (ANIDADO) ---
    for alpha in alpha_values:
        for h_dc in H_DC_values:
            start_time = time.time()
            print(f"Sim ({total_sim_count}/{total_sims}): alpha = {alpha:.3f}, h_dc/J = {h_dc/J:.3f} ...")
            
            # 1. Condición Inicial: FM Metaestable (Sz = +1)
            S_initial = np.zeros((N, 3))
            S_initial[:, 2] = 1.0
            
            # 2. Argumentos para el solver
            h_args = {'pulse_params': h_pulse_params, 'h_dc': h_dc}
            
            # 3. Resolver la dinámica (el corazón del trabajo)
            t_span = [0, T_MAX]
            t_eval = np.arange(0, T_MAX, DT_SAVE)
            
            sol = solve_ivp(
                llg_core.llg_rhs, t_span, S_initial.flatten(),
                method='RK45', t_eval=t_eval,
                args=(N, J, D, Da, alpha, gamma, h_total, h_args)
            )
            
            # 4. Guardar los resultados
            # Usamos un nombre de archivo sistemático
            alpha_str = f"{alpha:.3f}".replace('.', 'p')
            hdc_str = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
            output_filename = os.path.join(OUTPUT_DIR, f"datos_a{alpha_str}_h{hdc_str}.npz")
            
            np.savez_compressed(output_filename, S_history=sol.y, time_points=sol.t)
            
            end_time = time.time()
            print(f"  -> Sim. completada en {end_time - start_time:.1f} s. Datos guardados.")
            total_sim_count += 1

    print("--- BARRIDO COMPLETO DE MOVILIDAD FINALIZADO ---")
