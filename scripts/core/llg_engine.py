# --- llg_core.py ---
# Contiene todas las definiciones físicas del modelo 1D
# (Basado en 0_encontrar_estado_fundamental.py y 1_codigo_estatico.py)
import numpy as np

def calculate_B_eff(S, J, D, Da, h_external):
    """
    Calcula el campo efectivo B_eff en cada sitio de la cadena.
    S tiene forma (N, 3).
    h_external puede ser un vector (N, 3) o un escalar (0).
    """
    N = S.shape[0]
    
    # Condiciones de contorno periódicas
    s_next = np.roll(S, -1, axis=0)
    s_prev = np.roll(S, 1, axis=0)
    
    # 1. Intercambio de Heisenberg
    b_heisenberg = J * (s_prev + s_next)
    
    # 2. Interacción Dzyaloshinskii-Moriya (DMI)
    # Asumimos D = (0, 0, D_z) como en tu paper
    diff = s_next - s_prev
    b_dmi = np.zeros_like(S)
    b_dmi[:, 0] = -D * diff[:, 1]
    b_dmi[:, 1] =  D * diff[:, 0]
    
    # 3. Anisotropía de Eje Fácil (Da < 0)
    b_anisotropy = np.zeros_like(S)
    b_anisotropy[:, 2] = -2 * Da * S[:, 2]
    
    # Suma de todas las contribuciones
    return b_heisenberg + b_dmi + b_anisotropy + h_external

def find_ground_state(N, J, D, Da, max_steps=20000, tolerance=1e-8, dt_relax=0.05):
    """
    Encuentra el estado fundamental mediante relajación (descenso de gradiente).
    Inicia desde un estado aleatorio para evitar mínimos locales obvios.
    """
    # Condición Inicial: Estado Aleatorio
    S = np.random.rand(N, 3) - 0.5
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    
    for step in range(max_steps):
        # El campo externo es 0 para el estado fundamental estático
        B_eff = calculate_B_eff(S, J, D, Da, h_external=0)
        
        S_old = S.copy()
        
        # Alineamos los espines con el campo efectivo y renormalizamos
        # Esto es un descenso de gradiente en una esfera
        S += dt_relax * B_eff
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        
        # Verificación de convergencia
        if step % 500 == 0:
            change = np.max(np.abs(S - S_old))
            if change < tolerance:
                # print(f"  -> Convergencia en paso {step}.")
                return S # Estado fundamental encontrado

    # print("  -> Advertencia: Máximo de iteraciones alcanzado.")
    return S # Devuelve el mejor estado encontrado

def llg_rhs(t, S_flat, N, J, D, Da, alpha, gamma, h_external_func=None, h_args=None):
    """
    Calcula el lado derecho de la ecuación LLG (dS/dt) para el solver.
    """
    S = S_flat.reshape((N, 3))
    
    # Normalizar espines para estabilidad numérica
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Evitar división por cero
    S /= norms

    # Obtener el campo externo en este tiempo t
    if h_external_func:
        h_ext = h_external_func(t, N, **h_args)
    else:
        h_ext = 0.0 # Sin campo externo
        
    B_eff = calculate_B_eff(S, J, D, Da, h_ext)
    
    # Ecuación LLG
    precession_term = -gamma * np.cross(S, B_eff)
    damping_term = -alpha * np.cross(S, np.cross(S, B_eff))
    
    dSdt = precession_term + damping_term
    
    return dSdt.flatten() # Devolvemos un array plano
