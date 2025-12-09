# --- run_mapeo_fase.py ---
# TAREA 1: Generar sistemáticamente el diagrama de fase estático.
# Esto aborda la Crítica A y C del Réferi.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Importamos nuestro módulo de física desde la carpeta superior
# (Esto requiere que la carpeta raíz del proyecto esté en el PYTHONPATH)
# Por simplicidad, asumimos que están en la misma carpeta o que se ajusta el path:
# import sys
# sys.path.append('../01_modelo_fisico')
# import llg_core 
# 
# *** Inicio de la solución simple (copiar llg_core.py en esta carpeta) ***
# Para evitar problemas de importación, puedes simplemente copiar `llg_core.py`
# en esta misma carpeta `02_fase_estatica/`
import llg_core
# *** Fin de la solución simple ***


# --- 1. Parámetros de Simulación ---
N = 200
J = 1.0
OUTPUT_DIR = "mapeo_fase_Da_vs_D"

# --- 2. Definición de la Grilla de Exploración ---
# ¡Aquí está la sistematicidad que pidió el réferi!
# Una grilla densa, no solo 3 puntos.
D_values_over_J = np.linspace(0.1, 1.0, 20)  # 20 puntos para D/J
Da_values_over_J = np.linspace(0.0, -0.5, 20) # 20 puntos para Da/J (total 400 simulaciones)

# --- 3. Funciones de Visualización y Análisis ---
def plot_ground_state(S, J, D, Da, filename):
    """
    Grafica la configuración de espines Sx, Sy, Sz y la guarda.
    """
    plt.figure(figsize=(12, 7))
    sites = np.arange(N)
    plt.plot(sites, S[:, 0], 'o-', label='$S_x$', markersize=4, alpha=0.8)
    plt.plot(sites, S[:, 1], 's-', label='$S_y$', markersize=4, alpha=0.8)
    plt.plot(sites, S[:, 2], '^-', label='$S_z$', markersize=4, alpha=0.8)
    
    plt.xlabel('Sitio en la cadena ($i$)')
    plt.ylabel('Componente del Espín')
    title = f'Estado Fundamental: D/J={D/J:.3f}, Da/J={Da/J:.3f}'
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

def analyze_phase(S):
    """
    Analiza una configuración de espines y devuelve la fase.
    Esta es una función clave que debemos definir.
    """
    # Criterio simple para diferenciar fases (puedes refinarlo)
    sz_mean = np.mean(S[:, 2])
    sz_std = np.std(S[:, 2])
    
    if sz_mean > 0.95:
        # Casi perfectamente alineado -> Ferromagnético (Metaestable)
        return "FM" 
    elif sz_std < 0.6:
        # Olas anchas en Sz -> Red de Solitones (SL)
        # (La fase Helicoidal tiene sz_std=0)
        return "SL" 
    else:
        # Sz es casi cero, oscilaciones en Sx, Sy -> Helicoidal (H)
        return "H"

# --- 4. Ejecución del Barrido ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directorio '{OUTPUT_DIR}' creado.")

    print("--- INICIANDO MAPEO SISTEMÁTICO DE FASE ---")
    
    # Matriz para guardar los resultados del diagrama de fase
    phase_diagram = np.empty((len(Da_values_over_J), len(D_values_over_J)), dtype=object)

    for i, da_ratio in enumerate(Da_values_over_J):
        for j, d_ratio in enumerate(D_values_over_J):
            Da = da_ratio * J
            D = d_ratio * J
            
            print(f"Procesando (Da/J={da_ratio:.3f}, D/J={d_ratio:.3f}) ...")
            
            # 1. Encontrar estado fundamental (importado desde el core)
            S_ground = llg_core.find_ground_state(N, J, D, Da)
            
            # 2. Guardar la imagen de la configuración
            da_str = f"{da_ratio:.3f}".replace('.', 'p').replace('-', 'm')
            d_str = f"{d_ratio:.3f}".replace('.', 'p')
            filename = os.path.join(OUTPUT_DIR, f"fase_Da{da_str}_D{d_str}.png")
            plot_ground_state(S_ground, J, D, Da, filename)
            
            # 3. Analizar y guardar la fase
            phase_name = analyze_phase(S_ground)
            phase_diagram[i, j] = phase_name
            print(f"  -> Fase identificada: {phase_name}")

    print("--- MAPEO COMPLETO ---")
    
    # 4. Guardar la matriz del diagrama de fase
    np.save(os.path.join(OUTPUT_DIR, "diagrama_fase_matriz.npy"), phase_diagram)
    print("Matriz del diagrama de fase guardada en 'diagrama_fase_matriz.npy'")

    # (Aquí iría el código para graficar el diagrama de fase 2D,
    #  pero lo dejamos para un script de análisis separado)
