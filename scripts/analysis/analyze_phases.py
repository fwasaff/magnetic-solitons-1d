# --- analizar_fases.py ---
# TAREA: Cargar la matriz de datos generada por 'run_mapeo_fase.py'
# y generar el diagrama de fase 2D final (la "Nueva Figura 1").

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 1. Parámetros de Simulación (DEBEN COINCIDIR EXACTAMENTE) ---
# Estos valores deben ser idénticos a los de 'run_mapeo_fase.py'
D_values_over_J = np.linspace(0.1, 1.0, 20)
Da_values_over_J = np.linspace(0.0, -0.5, 20)
DATA_FILE = "diagrama_fase_matriz.npy"
OUTPUT_FIGURE = "figure_diagrama_fase.png"

# --- 2. Carga y Procesamiento de Datos ---
try:
    phase_data_str = np.load(DATA_FILE, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{DATA_FILE}'.")
    print("Por favor, ejecuta 'run_mapeo_fase.py' primero.")
    exit()

print(f"Archivo de datos '{DATA_FILE}' cargado con éxito.")
print(f"Dimensiones de los datos: {phase_data_str.shape}")

# --- 3. Mapeo de Fases (De String a Número) ---
# Convertimos los nombres ("H", "SL") a números para que matplotlib pueda graficarlos.
# 0 = Helicoidal (H)
# 1 = Red de Solitones (SL)
# 2 = Ferromagnético (FM)
phase_map_numeric = np.zeros(phase_data_str.shape, dtype=int)
phase_map_numeric[phase_data_str == 'H'] = 0
phase_map_numeric[phase_data_str == 'SL'] = 1
phase_map_numeric[phase_data_str == 'FM'] = 2

# --- 4. Visualización del Diagrama de Fase 2D ---
print("Generando el diagrama de fase 2D...")

# Definimos los bordes de los píxeles para pcolormesh
# Esto es más preciso que 'imshow' para los ejes.
D_edges = np.linspace(D_values_over_J[0], D_values_over_J[-1], len(D_values_over_J) + 1)
Da_edges = np.linspace(Da_values_over_J[0], Da_values_over_J[-1], len(Da_values_over_J) + 1)

# Creamos un mapa de colores discreto
# (Puedes cambiar 'blue', 'orange', 'green' por los colores que prefieras)
cmap = mcolors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(12, 10))
# Usamos pcolormesh para tener los ejes con los valores correctos
# Transponemos la matriz para que D/J esté en el eje X y Da/J en el Y
mesh = plt.pcolormesh(D_edges, Da_edges, phase_map_numeric, cmap=cmap, norm=norm, shading='auto')

# Configuración de la Barra de Color
cbar = plt.colorbar(mesh, ticks=[0, 1, 2])
cbar.set_ticklabels(['Helicoidal (H)', 'Red de Solitones (SL)', 'Ferromagnético (FM)'])
cbar.set_label('Fase del Estado Fundamental', rotation=270, labelpad=20)

# Etiquetas y Título
plt.xlabel(r'Interacción DMI ($D/J$)')
plt.ylabel(r'Anisotropía de Eje Fácil ($D_a/J$)')
plt.title('Diagrama de Fase Estático (Calculado Sistemáticamente)', fontsize=16, pad=20)

# Invertimos el eje Y para que los valores negativos (más anisotropía) estén abajo
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
print(f"¡Éxito! Diagrama de fase guardado como '{OUTPUT_FIGURE}'")
plt.show()
