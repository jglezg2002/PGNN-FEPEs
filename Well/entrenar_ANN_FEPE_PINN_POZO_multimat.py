#!/usr/bin/env python3
"""
entrenar_ANN_FEPE_PINN_POZO_multimat.py

PINN Nivel 2 para predecir FEPE(E, h, material) - detector pozo (GIRMA).

La red neuronal se entrena con FEPE del material de calibración (RGU-1,
datos PENELOPE/pozo) usando un subconjunto de alturas y energías.

Test en dos niveles, SOLO en alturas y energías NO vistas:
  (a) ANN sola: FEPE RGU-1 -> mide capacidad de interpolación pura.
  (b) Pipeline completo: para cada material de test (RGTh-1, agua, ...),
      ANN(E,h) × f_a vs ground truth PENELOPE de ese material.

Los materiales de test se auto-descubren buscando ficheros Excel
FEPE_barrido_alturas_POZO_*.xlsx (excluyendo el de entrenamiento).
Si un material no tiene composición definida en MATERIALES, se omite
la corrección f_a y se avisa para que el usuario la añada.

Corrección de autoatenuación por ray-tracing geométrico y parallel beam.
Ambos métodos se evalúan en paralelo para comparación.

Requiere: autoatenuacion_corregida_POZO.py en el mismo directorio.

Refs:
  - Barba-Lobo, Mosqueda, Bolívar (2021). Radiat. Phys. Chem. 179:109247
  - Berger, Hubbell et al. (2010). XCOM v1.5, NIST SRD 8

Autor: Jonay G. Guerra
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Corrección de autoatenuación por ray-tracing (geometría pozo) y parallel beam
from autoatenuacion_corregida_POZO import (
    CacheRayos, correccion_material_raytracing,
)


# ============================================================================
# ======================= PARÁMETROS CONFIGURABLES ==========================
# ============================================================================

# --- Definición de materiales ---
# Composición centesimal (fracciones en peso) y densidad PENELOPE.
# Las claves del diccionario deben coincidir con los nombres de columna
# del CSV XCOM: 'H_Z1', 'O_Z8', 'Si_Z14', etc.
#
# Solo los materiales que aparezcan aquí podrán ser evaluados con el
# pipeline completo (ANN × f_a). Los que falten se reportan con aviso.

MATERIALES = {
    'water': {
        'composicion': {'H_Z1': 0.1119, 'O_Z8': 0.8881},
        'densidad': 1.0,
    },
    'RGU_UNAM': {
        # IAEA-RGU-1. Fracciones en peso convertidas del .mat PENELOPE.
        'composicion': {
            'O_Z8': 0.5380, 'Na_Z11': 0.0010, 'Al_Z13': 0.0010,
            'Si_Z14': 0.4570, 'Ca_Z20': 0.0010, 'Fe_Z26': 0.0010,
            'U_Z92': 0.0010,
        },
        'densidad': 1.50,
    },
    'RGTh_UNAM': {
        # IAEA-RGTh-1. Fracciones en peso convertidas del .mat PENELOPE.
        'composicion': {
            'O_Z8': 0.535398, 'Mg_Z12': 0.0002, 'Al_Z13': 0.0004,
            'Si_Z14': 0.4500, 'P_Z15': 0.001290, 'Ca_Z20': 0.0042,
            'Mn_Z25': 0.000137, 'Fe_Z26': 0.0013, 'Ba_Z56': 0.006235,
            'Pb_Z82': 0.00003, 'Th_Z90': 0.000810,
        },
        'densidad': 2.00,
    },
    'RGK_UNAM': {
        # IAEA-RGK-1 (K2SO4). Fracciones en peso convertidas del .mat PENELOPE.
        'composicion': {
            'O_Z8': 0.367251, 'S_Z16': 0.184010, 'K_Z19': 0.448740,
        },
        'densidad': 1.75,
    },
    'ILMENITA': {
        # Ilmenita (muestra real). Fracciones en peso del .mat PENELOPE.
        'composicion': {
            'O_Z8': 0.374321, 'Mg_Z12': 0.001920, 'Al_Z13': 0.003759,
            'Si_Z14': 0.003967, 'S_Z16': 0.000040, 'Ca_Z20': 0.000357,
            'Ti_Z22': 0.297547, 'Mn_Z25': 0.010071, 'Fe_Z26': 0.308017,
        },
        'densidad': 2.70,
    },
    'FOSFOYESO_IAEA': {
        # Fosfoyeso IAEA. Fracciones en peso del .mat PENELOPE.
        'composicion': {
            'H_Z1': 0.020002, 'O_Z8': 0.560002,
            'S_Z16': 0.189995, 'Ca_Z20': 0.230001,
        },
        'densidad': 1.00,
    },
    'IAEA448': {
        # IAEA-448 (suelo). Fracciones en peso del .mat PENELOPE.
        'composicion': {
            'C_Z6': 0.1205, 'O_Z8': 0.455701, 'Na_Z11': 0.0073,
            'Mg_Z12': 0.0261, 'Al_Z13': 0.0416, 'Si_Z14': 0.1478,
            'S_Z16': 0.019799, 'K_Z19': 0.0083, 'Ca_Z20': 0.1049,
            'Fe_Z26': 0.067998,
        },
        'densidad': 1.30,
    },
    'IAEA326': {
        # IAEA-326 (suelo). Fracciones en peso del .mat PENELOPE.
        'composicion': {
            'C_Z6': 0.0955, 'O_Z8': 0.449501, 'Na_Z11': 0.0059,
            'Mg_Z12': 0.0062, 'Al_Z13': 0.0549, 'Si_Z14': 0.325001,
            'P_Z15': 0.0007, 'K_Z19': 0.0190, 'Ca_Z20': 0.0116,
            'Ti_Z22': 0.004499, 'Mn_Z25': 0.0006, 'Fe_Z26': 0.026599,
        },
        'densidad': 1.30,
    },
    'IAEA447': {
        # IAEA-447 (suelo/musgo). Fracciones en peso del .mat PENELOPE.
        'composicion': {
            'C_Z6': 0.4740, 'O_Z8': 0.1837, 'Na_Z11': 0.0031,
            'Mg_Z12': 0.0086, 'Al_Z13': 0.0449, 'Si_Z14': 0.1088,
            'K_Z19': 0.0171, 'Ca_Z20': 0.1261, 'Fe_Z26': 0.030399,
            'Tl_Z81': 0.0033,
        },
        'densidad': 1.10,
    },
}

# --- Mapeo de nombres internos -> nombres de publicación ---
# Las claves internas (coinciden con ficheros Excel) NO se modifican.
# Este diccionario se aplica SOLO en la capa de salida: figuras, JSON, consola.
NOMBRE_PUB = {
    'water':           'Water',
    'RGU_UNAM':        'IAEA-RGU-1',
    'RGTh_UNAM':       'IAEA-RGTh-1',
    'RGK_UNAM':        'IAEA-RGK-1',
    'ILMENITA':        'Ilmenite',
    'FOSFOYESO_IAEA':  'IAEA-434',
    'IAEA448':         'IAEA-448',
    'IAEA326':         'IAEA-326',
    'IAEA447':         'IAEA-447',
}


def nombre_pub(etiqueta):
    """Devuelve el nombre de publicación de un material, o la etiqueta tal cual."""
    return NOMBRE_PUB.get(etiqueta, etiqueta)


CONFIG = {
    # --- Datos de entrenamiento (material de calibración) ---
    # NOTA: los ficheros del barrido POZO siguen el patrón
    # FEPE_barrido_alturas_POZO_{etiqueta}.xlsx
    'fichero_train': 'FEPE_barrido_alturas_POZO_RGU_UNAM.xlsx',
    'material_train': 'RGU_UNAM',   # clave en MATERIALES
    'fichero_xcom': 'XCOM_mu_rho_table.csv',

    # --- Autodescubrimiento de materiales de test ---
    # Patrón glob para buscar excels del barrido.
    # La etiqueta del material se extrae del nombre de fichero:
    #   FEPE_barrido_alturas_POZO_{etiqueta}.xlsx -> etiqueta
    'patron_excels': 'FEPE_barrido_alturas_POZO_*.xlsx',

    # --- Preprocesamiento ---
    'log_E': True,

    # --- División train / test ---
    # Alturas usadas para entrenar (el resto se reserva para test)
    'alturas_train_cm': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1],
    # Energías reservadas para test (el resto se usa para entrenar)
    'energias_test_keV': [144.0, 185.96, 351.932, 768.356, 1001.0, 1120.285],
    'semilla_random': 21,
    'fraccion_validacion': 0.2,

    # --- Arquitectura ---
    'capas_ocultas': [64, 64, 32],
    'activacion': 'SiLU',

    # --- Normalización ---
    'normalizacion_X': 'minmax',
    'normalizacion_Y': 'log',

    # --- Entrenamiento ---
    'learning_rate': 1e-3,
    'epochs': 3000,
    'batch_size': 64,
    'weight_decay': 1e-5,
    'scheduler': True,
    'scheduler_patience': 50,
    'scheduler_factor': 0.5,
    'early_stopping': True,
    'early_stopping_patience': 500,

    # --- Salidas ---
    'guardar_modelo': True,
    'fichero_modelo': 'modelo_FEPE_PINN_POZO_multimat_extremos.pt',
    'directorio_figuras': 'figuras_ANN_PINN_POZO_multimat_extremos',
}


# ============================================================================
# =================== MÓDULO DE AUTOATENUACIÓN (CAPA ANALÍTICA) ==============
# ============================================================================

class TablaXCOM:
    """
    Carga la tabla XCOM y calcula μ/ρ para cualquier mezcla a cualquier energía.
    """

    def __init__(self, fichero_csv):
        df = pd.read_csv(fichero_csv)
        self.energias_keV = df['E_keV'].values
        self.energias_MeV = df['E_MeV'].values
        self.elementos = [c for c in df.columns if c not in ('E_MeV', 'E_keV')]
        self.tabla = df[self.elementos].values
        self.elem_to_idx = {e: i for i, e in enumerate(self.elementos)}
        self._avisos_emitidos = set()  # para no repetir avisos

    def mu_rho_mezcla(self, composicion, E_keV):
        E_keV = np.atleast_1d(np.float64(E_keV))
        mu_total = np.zeros_like(E_keV)
        for elem, w in composicion.items():
            if w == 0:
                continue
            if elem not in self.elem_to_idx:
                if elem not in self._avisos_emitidos:
                    print(f'  [!] XCOM: elemento {elem} (w={w:.4f}) no está '
                          f'en la tabla -> omitido del cálculo de μ/ρ')
                    self._avisos_emitidos.add(elem)
                continue
            ln_mu = np.interp(
                np.log(E_keV),
                np.log(self.energias_keV),
                np.log(self.tabla[:, self.elem_to_idx[elem]])
            )
            mu_total += w * np.exp(ln_mu)
        return mu_total


def factor_autoatenuacion_F(mu_rho, rho, h_cm):
    """F = [1 - exp(-μ·ρ·h)] / (μ·ρ·h)"""
    x = mu_rho * rho * h_cm
    return np.where(x > 1e-10, (1 - np.exp(-x)) / x, 1.0 - x / 2)


def correccion_material(xcom, comp_muestra, rho_muestra,
                        comp_patron, rho_patron, E_keV, h_cm):
    """f_a = F(muestra) / F(patrón) - haz paralelo."""
    E_keV = np.atleast_1d(np.float64(E_keV))
    h_cm = np.atleast_1d(np.float64(h_cm))
    mu_muestra = xcom.mu_rho_mezcla(comp_muestra, E_keV)
    mu_patron = xcom.mu_rho_mezcla(comp_patron, E_keV)
    F_muestra = factor_autoatenuacion_F(mu_muestra, rho_muestra, h_cm)
    F_patron = factor_autoatenuacion_F(mu_patron, rho_patron, h_cm)
    return F_muestra / F_patron


# ============================================================================
# =================== RED NEURONAL Y UTILIDADES ==============================
# ============================================================================

class Normalizador:
    def __init__(self, metodo='minmax'):
        self.metodo = metodo
        self.params = {}

    def ajustar_transformar(self, x):
        if self.metodo == 'minmax':
            self.params['min'] = x.min(axis=0)
            self.params['max'] = x.max(axis=0)
            rango = self.params['max'] - self.params['min']
            rango[rango == 0] = 1.0
            return (x - self.params['min']) / rango
        elif self.metodo == 'log':
            x_log = np.log(np.clip(x, 1e-12, None))
            self.params['min_log'] = x_log.min(axis=0)
            self.params['max_log'] = x_log.max(axis=0)
            rango = self.params['max_log'] - self.params['min_log']
            rango[rango == 0] = 1.0
            return (x_log - self.params['min_log']) / rango

    def transformar(self, x):
        if self.metodo == 'minmax':
            rango = self.params['max'] - self.params['min']
            rango[rango == 0] = 1.0
            return (x - self.params['min']) / rango
        elif self.metodo == 'log':
            x_log = np.log(np.clip(x, 1e-12, None))
            rango = self.params['max_log'] - self.params['min_log']
            rango[rango == 0] = 1.0
            return (x_log - self.params['min_log']) / rango

    def invertir(self, x_norm):
        if self.metodo == 'minmax':
            rango = self.params['max'] - self.params['min']
            return x_norm * rango + self.params['min']
        elif self.metodo == 'log':
            rango = self.params['max_log'] - self.params['min_log']
            x_log = x_norm * rango + self.params['min_log']
            return np.exp(x_log)


def get_activacion(nombre):
    return {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'SiLU': nn.SiLU(),
            'GELU': nn.GELU(), 'LeakyReLU': nn.LeakyReLU(0.01)}[nombre]


class MLP_FEPE(nn.Module):
    def __init__(self, capas_ocultas, activacion_nombre):
        super().__init__()
        capas = []
        n_in = 2
        for n in capas_ocultas:
            capas.append(nn.Linear(n_in, n))
            capas.append(get_activacion(activacion_nombre))
            n_in = n
        capas.append(nn.Linear(n_in, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


# ============================================================================
# =================== CARGA Y PREPARACIÓN DE DATOS ==========================
# ============================================================================

def cargar_datos(fichero):
    """Carga Excel del barrido PENELOPE: alturas, energías, matriz FEPE."""
    df = pd.read_excel(fichero)
    alturas = df.iloc[:, 0].values
    energias_keV = np.array([float(c.replace('_keV', '')) for c in df.columns[1:]])
    fepe_matrix = df.iloc[:, 1:].values
    return alturas, energias_keV, fepe_matrix


def buscar_indices(valores, lista_target, nombre='', tol=0.1):
    indices = []
    for v in lista_target:
        dist = np.abs(valores - v)
        idx = np.argmin(dist)
        if dist[idx] < tol:
            indices.append(idx)
        else:
            print(f'  [!] {nombre} {v} no encontrado (más cercano: {valores[idx]})')
    return np.array(sorted(indices))


def descubrir_materiales_test(cfg):
    """
    Busca ficheros Excel que coincidan con el patrón glob y extrae la
    etiqueta del material del nombre de fichero. Excluye el de train.

    FEPE_barrido_alturas_POZO_{etiqueta}.xlsx -> etiqueta
    """
    ficheros = sorted(glob.glob(cfg['patron_excels']))
    train_base = os.path.basename(cfg['fichero_train'])
    materiales_test = {}

    # Extraer etiqueta: FEPE_barrido_alturas_POZO_{etiqueta}.xlsx
    prefijo = 'FEPE_barrido_alturas_POZO_'
    sufijo = '.xlsx'

    for fpath in ficheros:
        fname = os.path.basename(fpath)
        if fname == train_base:
            continue
        # Excluir ficheros parciales del barrido
        if '_parcial' in fname:
            continue
        if fname.startswith(prefijo) and fname.endswith(sufijo):
            etiqueta = fname[len(prefijo):-len(sufijo)]
            materiales_test[etiqueta] = fpath

    return materiales_test


def preparar_datos(cfg, xcom):
    """
    Carga datos y prepara splits.

    Split:
      - Alturas train: cfg['alturas_train_cm']
      - Alturas test: todas las demás
      - Energías train: todas excepto cfg['energias_test_keV']
      - Energías test: cfg['energias_test_keV']

    Entrenamiento:  alturas_train × energías_train (solo RGU)
    Validación:     20% aleatorio del set de entrenamiento
    Test ANN:       alturas_test × energías_test (RGU, sin f_a)
    Test pipeline:  alturas_test × energías_test (cada material, con f_a)
    """
    mat_train_dict = MATERIALES[cfg['material_train']]

    # --- Cargar dataset de entrenamiento ---
    alturas, energias_keV, fepe_mat_train = cargar_datos(cfg['fichero_train'])

    print(f'\n  Fichero train: {cfg["fichero_train"]} ({nombre_pub(cfg["material_train"])})')
    print(f'  Malla: {len(alturas)} alturas × {len(energias_keV)} energías')

    # --- Autodescubrir materiales de test ---
    mat_test_ficheros = descubrir_materiales_test(cfg)
    print(f'\n  Materiales de test descubiertos:')

    datos_test_MC = {}
    for etiqueta, fpath in mat_test_ficheros.items():
        alt_te, ene_te, fepe_te = cargar_datos(fpath)
        assert np.allclose(alt_te, alturas), \
            f'Alturas de {fpath} no coinciden con train'
        assert np.allclose(ene_te, energias_keV), \
            f'Energías de {fpath} no coinciden con train'
        datos_test_MC[etiqueta] = fepe_te
        tiene_comp = etiqueta in MATERIALES
        print(f'    {etiqueta:12s} <- {fpath}  '
              f'{"[OK] composición" if tiene_comp else "[!] SIN composición (f_a omitido)"}')

    if len(datos_test_MC) == 0:
        print('  [!] No se encontraron materiales de test.')

    # --- Split alturas ---
    idx_h_train = buscar_indices(alturas, cfg['alturas_train_cm'],
                                 'Altura', tol=1e-3)
    mascara_h_test = np.ones(len(alturas), dtype=bool)
    mascara_h_test[idx_h_train] = False
    idx_h_test = np.where(mascara_h_test)[0]

    # --- Split energías ---
    idx_E_test = buscar_indices(energias_keV, cfg['energias_test_keV'],
                                'Energía', tol=0.1)
    mascara_E_train = np.ones(len(energias_keV), dtype=bool)
    mascara_E_train[idx_E_test] = False
    idx_E_train = np.where(mascara_E_train)[0]

    energias_train = energias_keV[idx_E_train]
    energias_test = energias_keV[idx_E_test]
    alturas_train = alturas[idx_h_train]
    alturas_test = alturas[idx_h_test]

    print(f'\n  Alturas train:  {len(idx_h_train):3d} -> '
          f'{[round(float(h), 1) for h in alturas_train]} cm')
    print(f'  Alturas test:   {len(idx_h_test):3d} (todas las demás)')
    print(f'  Energías train: {len(idx_E_train):3d} -> '
          f'{[round(float(e), 1) for e in energias_train]} keV')
    print(f'  Energías test:  {len(idx_E_test):3d} -> '
          f'{[round(float(e), 1) for e in energias_test]} keV')

    # --- Construir arrays de entrenamiento ---
    # Train/val: alturas_train × energías_train (FEPE RGU)
    E_tr, h_tr, fepe_tr = [], [], []
    for ih in idx_h_train:
        for ie in idx_E_train:
            E_tr.append(energias_keV[ie])
            h_tr.append(alturas[ih])
            fepe_tr.append(fepe_mat_train[ih, ie])
    E_tr = np.array(E_tr)
    h_tr = np.array(h_tr)
    fepe_tr = np.array(fepe_tr)

    # --- Construir arrays de test ---
    # Test: alturas_test × energías_test (puntos NO vistos en ningún eje)
    E_te, h_te = [], []
    fepe_te_train_mat = []
    for ih in idx_h_test:
        for ie in idx_E_test:
            E_te.append(energias_keV[ie])
            h_te.append(alturas[ih])
            fepe_te_train_mat.append(fepe_mat_train[ih, ie])
    E_te = np.array(E_te)
    h_te = np.array(h_te)
    fepe_te_train_mat = np.array(fepe_te_train_mat)

    # Ground truth de cada material de test en los mismos puntos
    fepe_te_MC_por_mat = {}
    for etiqueta, fepe_matrix in datos_test_MC.items():
        fepe_te = []
        for ih in idx_h_test:
            for ie in idx_E_test:
                fepe_te.append(fepe_matrix[ih, ie])
        fepe_te_MC_por_mat[etiqueta] = np.array(fepe_te)

    # --- Precalcular correcciones f_a (ray-tracing) por material ---
    materiales_con_fa = [k for k in datos_test_MC if k in MATERIALES]
    cache_rayos = None

    if len(materiales_con_fa) > 0:
        print('\n  Precalculando rayos para corrección de ángulo sólido...')
        cache_rayos = CacheRayos(N_source=8000, N_det=400)
        alturas_unicas_test = np.unique(np.round(h_te, 3))
        t_rayos = time.time()
        for h_u in alturas_unicas_test:
            cache_rayos.get(h_u)
        print(f'    {len(alturas_unicas_test)} alturas precalculadas en '
              f'{time.time() - t_rayos:.1f} s')

    f_a_por_mat = {}
    f_a_pb_por_mat = {}
    f_a_MC_por_mat = {}

    for etiqueta in materiales_con_fa:
        mat_test_dict = MATERIALES[etiqueta]

        # Ray-tracing
        f_a_rt = correccion_material_raytracing(
            xcom,
            comp_muestra=mat_test_dict['composicion'],
            rho_muestra=mat_test_dict['densidad'],
            comp_patron=mat_train_dict['composicion'],
            rho_patron=mat_train_dict['densidad'],
            E_keV=E_te, h_cm=h_te,
            cache=cache_rayos,
        )
        f_a_por_mat[etiqueta] = f_a_rt

        # Haz paralelo (para comparación)
        f_a_pb = np.array([
            correccion_material(
                xcom,
                comp_muestra=mat_test_dict['composicion'],
                rho_muestra=mat_test_dict['densidad'],
                comp_patron=mat_train_dict['composicion'],
                rho_patron=mat_train_dict['densidad'],
                E_keV=E_te[i], h_cm=h_te[i]
            ).item()
            for i in range(len(E_te))
        ])
        f_a_pb_por_mat[etiqueta] = f_a_pb

        # f_a "real" de Monte Carlo
        f_a_MC = fepe_te_MC_por_mat[etiqueta] / fepe_te_train_mat
        f_a_MC_por_mat[etiqueta] = f_a_MC

        # Resumen
        err_fa_rt = np.abs(f_a_rt - f_a_MC) / \
                    np.clip(np.abs(f_a_MC), 1e-12, None) * 100
        err_fa_pb = np.abs(f_a_pb - f_a_MC) / \
                    np.clip(np.abs(f_a_MC), 1e-12, None) * 100
        print(f'\n  f_a {nombre_pub(cfg["material_train"])}->{nombre_pub(etiqueta)}:')
        print(f'    Rango RT:  {f_a_rt.min():.4f} - {f_a_rt.max():.4f}  '
              f'(error medio vs MC: {np.mean(err_fa_rt):.2f}%)')
        print(f'    Rango PB:  {f_a_pb.min():.4f} - {f_a_pb.max():.4f}  '
              f'(error medio vs MC: {np.mean(err_fa_pb):.2f}%)')
        print(f'    Rango MC:  {f_a_MC.min():.4f} - {f_a_MC.max():.4f}')

    # --- log(E) si procede ---
    E_tr_proc = np.log(E_tr) if cfg.get('log_E') else E_tr.copy()
    E_te_proc = np.log(E_te) if cfg.get('log_E') else E_te.copy()

    # --- Separar validación ---
    n_total = len(E_tr)
    n_val = int(n_total * cfg['fraccion_validacion'])
    np.random.seed(cfg['semilla_random'])
    perm = np.random.permutation(n_total)

    X_all = np.column_stack([E_tr_proc, h_tr])
    Y_all = fepe_tr.reshape(-1, 1)

    X_train = X_all[perm[n_val:]]
    Y_train = Y_all[perm[n_val:]]
    X_val = X_all[perm[:n_val]]
    Y_val = Y_all[perm[:n_val]]
    X_test = np.column_stack([E_te_proc, h_te])

    # --- Normalización ---
    norm_X = Normalizador(cfg['normalizacion_X'])
    norm_Y = Normalizador(cfg['normalizacion_Y'])
    X_train_n = norm_X.ajustar_transformar(X_train)
    X_val_n = norm_X.transformar(X_val)
    X_test_n = norm_X.transformar(X_test)
    Y_train_n = norm_Y.ajustar_transformar(Y_train)
    Y_val_n = norm_Y.transformar(Y_val)

    datos = {
        # Arrays normalizados para PyTorch
        'X_train': X_train_n, 'Y_train': Y_train_n,
        'X_val': X_val_n, 'Y_val': Y_val_n,
        'X_test': X_test_n,
        # Ground truth
        'Y_test_train_mat': fepe_te_train_mat,
        'fepe_te_MC_por_mat': fepe_te_MC_por_mat,
        # Correcciones
        'f_a_por_mat': f_a_por_mat,
        'f_a_pb_por_mat': f_a_pb_por_mat,
        'f_a_MC_por_mat': f_a_MC_por_mat,
        'cache_rayos': cache_rayos,
        # Metadatos de la malla
        'E_test_keV': np.exp(E_te_proc) if cfg.get('log_E') else E_te,
        'h_test_cm': h_te,
        'norm_X': norm_X, 'norm_Y': norm_Y,
        'alturas': alturas,
        'alturas_train': alturas_train, 'alturas_test': alturas_test,
        'energias_keV': energias_keV,
        'energias_train': energias_train, 'energias_test': energias_test,
        'idx_h_train': idx_h_train, 'idx_h_test': idx_h_test,
        'idx_E_train': idx_E_train, 'idx_E_test': idx_E_test,
        'fepe_matrix_train': fepe_mat_train,
        'datos_test_MC_matrices': datos_test_MC,
    }

    print(f'\nResumen de datos:')
    print(f'  Train:      {len(X_train):4d} pts  '
          f'({nombre_pub(cfg["material_train"])}, {len(idx_h_train)} alturas × '
          f'{len(idx_E_train)} energías)')
    print(f'  Validación: {len(X_val):4d} pts')
    print(f'  Test:       {len(X_test):4d} pts  '
          f'({len(idx_h_test)} alturas × {len(idx_E_test)} energías)')
    print(f'  Materiales test con f_a: '
          f'{[nombre_pub(k) for k in f_a_por_mat]}')
    mat_sin_comp = [k for k in datos_test_MC if k not in MATERIALES]
    if mat_sin_comp:
        print(f'  Materiales test SIN composición (f_a omitido): '
              f'{[nombre_pub(k) for k in mat_sin_comp]}')

    return datos


# ============================================================================
# =================== ENTRENAMIENTO =========================================
# ============================================================================

def entrenar(cfg, datos):
    # Fijar semilla de PyTorch para inicialización de pesos reproducible
    torch.manual_seed(cfg['semilla_random'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['semilla_random'])

    X_train_t = torch.FloatTensor(datos['X_train'])
    Y_train_t = torch.FloatTensor(datos['Y_train'])
    X_val_t = torch.FloatTensor(datos['X_val'])
    Y_val_t = torch.FloatTensor(datos['Y_val'])

    train_ds = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)

    modelo = MLP_FEPE(cfg['capas_ocultas'], cfg['activacion'])
    print(f'\nArquitectura: {modelo}')
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f'Parámetros entrenables: {n_params}')

    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'])
    scheduler = None
    if cfg['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizador, mode='min', patience=cfg['scheduler_patience'],
            factor=cfg['scheduler_factor'], min_lr=1e-7)

    criterio = nn.MSELoss()
    hist = {'train_loss': [], 'val_loss': [], 'lr': []}
    mejor_val_loss = float('inf')
    mejor_estado = None
    epochs_sin_mejora = 0
    t0 = time.time()

    for epoch in range(cfg['epochs']):
        modelo.train()
        perdida_epoch = 0.0
        n_batches = 0
        for X_batch, Y_batch in train_loader:
            optimizador.zero_grad()
            loss = criterio(modelo(X_batch), Y_batch)
            loss.backward()
            optimizador.step()
            perdida_epoch += loss.item()
            n_batches += 1
        train_loss = perdida_epoch / n_batches

        modelo.eval()
        with torch.no_grad():
            val_loss = criterio(modelo(X_val_t), Y_val_t).item()

        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)
        hist['lr'].append(optimizador.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
        if cfg['early_stopping'] and \
                epochs_sin_mejora >= cfg['early_stopping_patience']:
            print(f'\nEarly stopping en epoch {epoch + 1}')
            break

        intervalo = max(cfg['epochs'] // 10, 1)
        if (epoch + 1) % intervalo == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:5d}/{cfg["epochs"]}  '
                  f'train={train_loss:.2e}  val={val_loss:.2e}  '
                  f'lr={optimizador.param_groups[0]["lr"]:.1e}')

    print(f'\nEntrenamiento: {time.time() - t0:.1f} s  '
          f'({len(hist["train_loss"])} epochs)')
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        print(f'Restaurado mejor modelo (val_loss = {mejor_val_loss:.2e})')
    return modelo, hist


# ============================================================================
# =================== EVALUACIÓN =============================================
# ============================================================================

def calcular_metricas(y_true, y_pred, nombre=''):
    """Calcula MAPE, R², max_err para un par de arrays."""
    residuos = y_true - y_pred
    err_rel = np.abs(residuos) / np.clip(np.abs(y_true), 1e-12, None) * 100
    mape = float(np.mean(err_rel))
    ss_res = np.sum(residuos ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    max_err = float(np.max(err_rel))
    return {'MAPE_%': mape, 'R2': r2, 'max_err_%': max_err, 'nombre': nombre}


def evaluar(cfg, modelo, datos):
    """
    Evaluación completa:
      1. ANN sola -> predice FEPE RGU en puntos test (interpola E y h)
      2. Pipeline por material -> ANN × f_a vs PENELOPE
      3. Métricas por energía para cada material
    """
    norm_Y = datos['norm_Y']
    n_h_test = len(datos['idx_h_test'])
    n_E_test = len(datos['energias_test'])

    # --- 1. ANN predice FEPE del material de entrenamiento ---
    modelo.eval()
    with torch.no_grad():
        Y_pred_train_mat = norm_Y.invertir(
            modelo(torch.FloatTensor(datos['X_test'])).numpy()
        ).flatten()

    Y_true_train_mat = datos['Y_test_train_mat']

    metricas_ann = calcular_metricas(Y_true_train_mat, Y_pred_train_mat,
                                     f'ANN sola ({nombre_pub(cfg["material_train"])})')

    print(f'\n{"=" * 60}')
    print(f'TEST - ANN sola (FEPE {nombre_pub(cfg["material_train"])}, '
          f'alturas y energías NO vistas):')
    print(f'  MAPE:     {metricas_ann["MAPE_%"]:.2f} %')
    print(f'  R²:       {metricas_ann["R2"]:.6f}')
    print(f'  Max err:  {metricas_ann["max_err_%"]:.2f} %')

    # --- 2. Pipeline por material ---
    resultados_por_mat = {}

    for etiqueta, fepe_MC in datos['fepe_te_MC_por_mat'].items():

        tiene_fa = etiqueta in datos['f_a_por_mat']

        if tiene_fa:
            f_a_rt = datos['f_a_por_mat'][etiqueta]
            f_a_pb = datos['f_a_pb_por_mat'][etiqueta]
            f_a_MC = datos['f_a_MC_por_mat'][etiqueta]

            # Pipeline con ray-tracing
            Y_pred_mat_rt = Y_pred_train_mat * f_a_rt
            metricas_pipe_rt = calcular_metricas(fepe_MC, Y_pred_mat_rt,
                                                  f'Pipeline RT -> {nombre_pub(etiqueta)}')

            # Pipeline con parallel beam
            Y_pred_mat_pb = Y_pred_train_mat * f_a_pb
            metricas_pipe_pb = calcular_metricas(fepe_MC, Y_pred_mat_pb,
                                                  f'Pipeline PB -> {nombre_pub(etiqueta)}')

            print(f'\nTEST - Pipeline vs PENELOPE {nombre_pub(etiqueta)} '
                  f'[well det.]:')
            print(f'  Ray-tracing:    MAPE = {metricas_pipe_rt["MAPE_%"]:.2f}%,  '
                  f'R² = {metricas_pipe_rt["R2"]:.6f},  '
                  f'max err = {metricas_pipe_rt["max_err_%"]:.2f}%')
            print(f'  Parallel beam:  MAPE = {metricas_pipe_pb["MAPE_%"]:.2f}%,  '
                  f'R² = {metricas_pipe_pb["R2"]:.6f},  '
                  f'max err = {metricas_pipe_pb["max_err_%"]:.2f}%')

            # Métricas por energía
            Y_true_2d = fepe_MC.reshape(n_h_test, n_E_test)
            Y_pred_rt_2d = Y_pred_mat_rt.reshape(n_h_test, n_E_test)
            Y_pred_pb_2d = Y_pred_mat_pb.reshape(n_h_test, n_E_test)
            f_a_rt_2d = f_a_rt.reshape(n_h_test, n_E_test)
            f_a_pb_2d = f_a_pb.reshape(n_h_test, n_E_test)
            f_a_MC_2d = f_a_MC.reshape(n_h_test, n_E_test)

            metricas_E = []
            print(f'\n  Error por energía (Pipeline vs PENELOPE {nombre_pub(etiqueta)}):')
            for j, ek in enumerate(datos['energias_test']):
                col_r = Y_true_2d[:, j]
                col_p_rt = Y_pred_rt_2d[:, j]
                col_p_pb = Y_pred_pb_2d[:, j]
                m_j_rt = calcular_metricas(col_r, col_p_rt)
                m_j_pb = calcular_metricas(col_r, col_p_pb)

                f_a_rt_medio = f_a_rt_2d[:, j].mean()
                f_a_pb_medio = f_a_pb_2d[:, j].mean()
                f_a_MC_medio = f_a_MC_2d[:, j].mean()

                err_fa_rt_j = np.abs(f_a_rt_2d[:, j] - f_a_MC_2d[:, j]) / \
                              np.clip(np.abs(f_a_MC_2d[:, j]), 1e-12, None) * 100
                err_fa_pb_j = np.abs(f_a_pb_2d[:, j] - f_a_MC_2d[:, j]) / \
                              np.clip(np.abs(f_a_MC_2d[:, j]), 1e-12, None) * 100

                print(f'    {ek:10.1f} keV:  '
                      f'MAPE RT={m_j_rt["MAPE_%"]:.2f}% PB={m_j_pb["MAPE_%"]:.2f}%,  '
                      f'R² RT={m_j_rt["R2"]:.4f} PB={m_j_pb["R2"]:.4f},  '
                      f'f_a(RT/PB/MC) = {f_a_rt_medio:.4f}/'
                      f'{f_a_pb_medio:.4f}/{f_a_MC_medio:.4f},  '
                      f'err f_a RT={np.mean(err_fa_rt_j):.2f}% '
                      f'PB={np.mean(err_fa_pb_j):.2f}%')

                metricas_E.append({
                    'energia_keV': round(float(ek), 3),
                    'MAPE_RT_%': round(m_j_rt['MAPE_%'], 3),
                    'MAPE_PB_%': round(m_j_pb['MAPE_%'], 3),
                    'R2_RT': round(m_j_rt['R2'], 6),
                    'R2_PB': round(m_j_pb['R2'], 6),
                    'f_a_raytracing': round(float(f_a_rt_medio), 4),
                    'f_a_parallel_beam': round(float(f_a_pb_medio), 4),
                    'f_a_MC': round(float(f_a_MC_medio), 4),
                    'error_f_a_RT_%': round(float(np.mean(err_fa_rt_j)), 3),
                    'error_f_a_PB_%': round(float(np.mean(err_fa_pb_j)), 3),
                })

            resultados_por_mat[etiqueta] = {
                'Y_pred_mat_rt': Y_pred_mat_rt,
                'Y_pred_mat_pb': Y_pred_mat_pb,
                'Y_true_MC': fepe_MC,
                'metricas_rt': metricas_pipe_rt,
                'metricas_pb': metricas_pipe_pb,
                'metricas_por_energia': metricas_E,
                'f_a_rt': f_a_rt,
                'f_a_pb': f_a_pb,
                'f_a_MC': f_a_MC,
                'Y_true_2d': Y_true_2d,
                'Y_pred_rt_2d': Y_pred_rt_2d,
                'Y_pred_pb_2d': Y_pred_pb_2d,
            }
        else:
            print(f'\n  [!] {etiqueta}: sin composición en MATERIALES -> '
                  f'pipeline f_a omitido. Añade la composición y densidad '
                  f'al diccionario MATERIALES para incluirlo.')

    print(f'{"=" * 60}')

    return {
        'metricas_ann': metricas_ann,
        'Y_pred_train_mat': Y_pred_train_mat,
        'Y_true_train_mat': Y_true_train_mat,
        'resultados_por_mat': resultados_por_mat,
    }


# ============================================================================
# =================== GRÁFICAS ==============================================
# ============================================================================

def generar_graficas(cfg, hist, datos, resultados, xcom):
    fig_dir = Path(cfg['directorio_figuras'])
    fig_dir.mkdir(exist_ok=True)

    mat_train_nombre = nombre_pub(cfg['material_train'])
    n_h_test = len(datos['idx_h_test'])
    n_E_test = len(datos['energias_test'])

    # ---- 1. Curvas de pérdida ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(hist['train_loss'], label='Train', alpha=0.8)
    ax.semilogy(hist['val_loss'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Loss evolution - well det. '
                 f'(ANN trained on {mat_train_nombre})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / '01_curvas_perdida.png', dpi=150)
    plt.close(fig)

    # ---- 2. Pred vs real: ANN sola (material train, pts no vistos) ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    Y_r = resultados['Y_true_train_mat']
    Y_p = resultados['Y_pred_train_mat']
    ax.scatter(Y_r, Y_p, s=10, alpha=0.5, edgecolors='none')
    lims = [min(Y_r.min(), Y_p.min()), max(Y_r.max(), Y_p.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel(f'FEPE {mat_train_nombre} (PENELOPE)')
    ax.set_ylabel(f'FEPE {mat_train_nombre} (ANN)')
    m_ann = resultados['metricas_ann']
    ax.set_title(f'ANN alone (unseen E and h) -> R²={m_ann["R2"]:.5f}, '
                 f'MAPE={m_ann["MAPE_%"]:.2f}%')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / '02_pred_vs_real_ANN_sola.png', dpi=150)
    plt.close(fig)

    # ---- 3. Por cada material de test: pred vs real + FEPE(h) + MAPE ----
    colores_mat = plt.cm.tab10.colors
    mat_list = sorted(resultados['resultados_por_mat'].keys())

    for i_mat, mat_nombre in enumerate(mat_list):
        res_mat = resultados['resultados_por_mat'][mat_nombre]
        color = colores_mat[i_mat % len(colores_mat)]
        mat_pub = nombre_pub(mat_nombre)

        # 3a. Pred vs real pipeline (RT y PB)
        m_pipe_rt = res_mat['metricas_rt']
        m_pipe_pb = res_mat['metricas_pb']

        fig, axes_pr = plt.subplots(1, 2, figsize=(12, 6))
        Y_r = res_mat['Y_true_MC']
        for i_corr, (Y_p, m_pipe, tag) in enumerate([
            (res_mat['Y_pred_mat_rt'], m_pipe_rt, 'RT'),
            (res_mat['Y_pred_mat_pb'], m_pipe_pb, 'PB'),
        ]):
            ax = axes_pr[i_corr]
            ax.scatter(Y_r, Y_p, s=10, alpha=0.5, edgecolors='none',
                       color=color)
            lims = [min(Y_r.min(), Y_p.min()), max(Y_r.max(), Y_p.max())]
            ax.plot(lims, lims, 'r--', linewidth=1)
            ax.set_xlabel(f'FEPE {mat_pub} (PENELOPE)')
            ax.set_ylabel(f'FEPE {mat_pub} (ANN × f_a {tag})')
            ax.set_title(f'{tag}: R²={m_pipe["R2"]:.5f}, '
                         f'MAPE={m_pipe["MAPE_%"]:.2f}%')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Pipeline -> {mat_pub}', fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / f'03_{mat_pub}_pred_vs_real.png', dpi=150)
        plt.close(fig)

        # 3b. FEPE(h) para cada energía de test
        n_cols = min(3, n_E_test)
        n_rows = (n_E_test + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for j, ek in enumerate(datos['energias_test']):
            ax = axes_flat[j]
            ax.plot(datos['alturas_test'],
                    res_mat['Y_true_2d'][:, j],
                    'k-', linewidth=1.2, label=f'{mat_pub} (PENELOPE)')
            ax.plot(datos['alturas_test'],
                    res_mat['Y_pred_rt_2d'][:, j],
                    'r--', linewidth=1.2, label=f'{mat_pub} (Pipeline-RT)')
            ax.plot(datos['alturas_test'],
                    res_mat['Y_pred_pb_2d'][:, j],
                    'b:', linewidth=1.2, label=f'{mat_pub} (Pipeline-PB)')

            m = res_mat['metricas_por_energia'][j]
            ax.set_xlabel('Height (cm)')
            ax.set_ylabel('FEPE')
            ax.set_title(f'{ek:.1f} keV - '
                         f'f_a: {m["f_a_raytracing"]:.3f} (RT) / '
                         f'{m["f_a_parallel_beam"]:.3f} (PB) / '
                         f'{m["f_a_MC"]:.3f} (MC)\n'
                         f'MAPE: RT={m["MAPE_RT_%"]:.2f}%  '
                         f'PB={m["MAPE_PB_%"]:.2f}%')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)

        for j_extra in range(n_E_test, len(axes_flat)):
            axes_flat[j_extra].set_visible(False)

        fig.suptitle(f'FEPE(h): Pipeline vs PENELOPE {mat_pub} - '
                     f'well det. - unseen E and h',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(fig_dir / f'04_{mat_pub}_FEPE_vs_h.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3c. MAPE por energía (RT vs PB)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        labels = [f'{m["energia_keV"]:.0f}'
                  for m in res_mat['metricas_por_energia']]
        mapes_rt = [m['MAPE_RT_%'] for m in res_mat['metricas_por_energia']]
        mapes_pb = [m['MAPE_PB_%'] for m in res_mat['metricas_por_energia']]
        x_pos = np.arange(len(labels))
        w = 0.35
        ax.bar(x_pos - w/2, mapes_rt, w, color=color, edgecolor='black',
               alpha=0.8, label='RT')
        ax.bar(x_pos + w/2, mapes_pb, w, color=color, edgecolor='black',
               alpha=0.4, label='PB')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel(f'Energy (keV)')
        ax.set_ylabel('MAPE (%)')
        ax.set_title(f'Error per energy - pipeline -> {mat_pub} '
                     f'(Overall MAPE: RT={m_pipe_rt["MAPE_%"]:.2f}%, '
                     f'PB={m_pipe_pb["MAPE_%"]:.2f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(fig_dir / f'05_{mat_pub}_MAPE_por_energia.png',
                    dpi=150)
        plt.close(fig)

    # ---- 4. Factor de autoatenuación vs E para all materials ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    E_all = datos['energias_keV']
    h_ejemplo = 2.5  # cm
    mat_train_dict = MATERIALES[cfg['material_train']]
    cache_rt = datos.get('cache_rayos')

    for i_mat, mat_nombre in enumerate(mat_list):
        if mat_nombre not in MATERIALES:
            continue
        mat_test_dict = MATERIALES[mat_nombre]
        color = colores_mat[i_mat % len(colores_mat)]
        mat_pub_i = nombre_pub(mat_nombre)

        # f_a haz paralelo
        f_a_vs_E_pb = []
        for E in E_all:
            fa = correccion_material(
                xcom, mat_test_dict['composicion'], mat_test_dict['densidad'],
                mat_train_dict['composicion'], mat_train_dict['densidad'],
                E, h_ejemplo
            ).item()
            f_a_vs_E_pb.append(fa)
        ax.semilogx(E_all, f_a_vs_E_pb, '-', color=color, linewidth=1.5,
             label=f'{mat_pub_i} (PB)')

        # f_a ray-tracing
        if cache_rt is not None:
            f_a_vs_E_rt = []
            for E in E_all:
                fa_rt = correccion_material_raytracing(
                    xcom, mat_test_dict['composicion'],
                    mat_test_dict['densidad'],
                    mat_train_dict['composicion'],
                    mat_train_dict['densidad'],
                    np.array([E]), np.array([h_ejemplo]), cache=cache_rt
                ).item()
                f_a_vs_E_rt.append(fa_rt)
            ax.semilogx(E_all, f_a_vs_E_rt, '--', color=color,
                         linewidth=1.5, label=f'{mat_pub_i} (RT)')

        # f_a Monte Carlo
        if mat_nombre in datos['datos_test_MC_matrices']:
            fepe_test_full = datos['datos_test_MC_matrices'][mat_nombre]
            idx_h_25 = np.argmin(np.abs(datos['alturas'] - h_ejemplo))
            f_a_MC_vs_E = fepe_test_full[idx_h_25, :] / \
                          datos['fepe_matrix_train'][idx_h_25, :]
            ax.semilogx(E_all, f_a_MC_vs_E, 'o', color=color,
                         markersize=5, label=f'{mat_pub_i} (MC)')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel(f'f_a (material / {nombre_pub(cfg["material_train"])})')
    ax.set_title(f'Self-attenuation factor: PB vs RT vs MC '
                 f'(h = {h_ejemplo} cm, well det.)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / '06_factor_autoatenuacion_todos.png', dpi=150)
    plt.close(fig)

    # ---- 5. Resumen comparativo: MAPE por material (RT vs PB) ----
    if len(mat_list) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(max(6, 2 * len(mat_list)), 5))
        nombres = []
        mapes_rt = []
        mapes_pb = []
        for mat_nombre in mat_list:
            nombres.append(nombre_pub(mat_nombre))
            mapes_rt.append(
                resultados['resultados_por_mat'][mat_nombre]['metricas_rt']['MAPE_%'])
            mapes_pb.append(
                resultados['resultados_por_mat'][mat_nombre]['metricas_pb']['MAPE_%'])
        x_pos = np.arange(len(nombres))
        w = 0.35
        bars_rt = ax.bar(x_pos - w/2, mapes_rt, w,
                         color=[colores_mat[i % len(colores_mat)]
                                for i in range(len(nombres))],
                         edgecolor='black', alpha=0.8, label='RT')
        bars_pb = ax.bar(x_pos + w/2, mapes_pb, w,
                         color=[colores_mat[i % len(colores_mat)]
                                for i in range(len(nombres))],
                         edgecolor='black', alpha=0.4, label='PB')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(nombres)
        ax.set_ylabel('Overall MAPE (%)')
        ax.set_title('Pipeline MAPE per test material '
                     '(unseen E and h) - RT vs PB')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars_rt, mapes_rt):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars_pb, mapes_pb):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / '07_resumen_MAPE_por_material.png', dpi=150)
        plt.close(fig)

    print(f'\nGráficas guardadas en {fig_dir}/')


# ============================================================================
# ========== GRÁFICAS RESUMEN - TODOS LOS MATERIALES COMBINADOS =============
# ============================================================================

def generar_graficas_resumen(cfg, datos, resultados):
    """
    Genera gráficas resumen que combinan TODOS los materiales de test:
      08 - Pred vs real pooled (all materials en un scatter)
      09 - FEPE(h) por energía con all materials superpuestos
      10 - MAPE por energía promediado sobre materiales (con dispersión)
      11 - Heatmap material × energía (MAPE)
      12 - Relative residuals vs FEPE (diagnóstico)
    """
    fig_dir = Path(cfg['directorio_figuras'])
    fig_dir.mkdir(exist_ok=True)

    n_h_test = len(datos['idx_h_test'])
    n_E_test = len(datos['energias_test'])
    mat_list = sorted(resultados['resultados_por_mat'].keys())

    if len(mat_list) == 0:
        print('  No hay materiales de test -> sin gráficas resumen.')
        return

    colores_mat = plt.cm.tab10.colors
    color_map = {m: colores_mat[i % len(colores_mat)]
                 for i, m in enumerate(mat_list)}

    det_label = 'pozo'

    # ---- 08. Pred vs real POOLED: all materials ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    all_true_rt, all_pred_rt = [], []
    all_true_pb, all_pred_pb = [], []

    for mat_nombre in mat_list:
        res = resultados['resultados_por_mat'][mat_nombre]
        c = color_map[mat_nombre]
        for i_corr, (tag, Y_p_key, ax) in enumerate([
            ('RT', 'Y_pred_mat_rt', axes[0]),
            ('PB', 'Y_pred_mat_pb', axes[1]),
        ]):
            Y_r = res['Y_true_MC']
            Y_p = res[Y_p_key]
            ax.scatter(Y_r, Y_p, s=8, alpha=0.45, edgecolors='none',
                       color=c, label=nombre_pub(mat_nombre))
            if tag == 'RT':
                all_true_rt.extend(Y_r)
                all_pred_rt.extend(Y_p)
            else:
                all_true_pb.extend(Y_r)
                all_pred_pb.extend(Y_p)

    # Métricas globales pooled
    m_pool_rt = calcular_metricas(np.array(all_true_rt),
                                   np.array(all_pred_rt), 'pooled RT')
    m_pool_pb = calcular_metricas(np.array(all_true_pb),
                                   np.array(all_pred_pb), 'pooled PB')

    for ax, m_pool, tag in [(axes[0], m_pool_rt, 'RT'),
                             (axes[1], m_pool_pb, 'PB')]:
        all_vals = np.concatenate([np.array(all_true_rt if tag == 'RT'
                                            else all_true_pb),
                                   np.array(all_pred_rt if tag == 'RT'
                                            else all_pred_pb)])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel(f'FEPE (PENELOPE)')
        ax.set_ylabel(f'FEPE (Pipeline × f_a {tag})')
        ax.set_title(f'{tag}: R²={m_pool["R2"]:.5f}, '
                     f'MAPE={m_pool["MAPE_%"]:.2f}%')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(f'Pipeline -> all materials - det. {det_label} '
                 f'(unseen E and h, {len(mat_list)} materiales)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / '08_RESUMEN_pred_vs_real_todos.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Guardar métricas pooled para JSON
    resultados['metricas_pooled_rt'] = m_pool_rt
    resultados['metricas_pooled_pb'] = m_pool_pb

    # ---- 09. FEPE(h) por energía - all materials superpuestos ----
    n_cols = min(3, n_E_test)
    n_rows = (n_E_test + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for j, ek in enumerate(datos['energias_test']):
        ax = axes_flat[j]
        mapes_rt_this_E = []
        mapes_pb_this_E = []
        for mat_nombre in mat_list:
            res = resultados['resultados_por_mat'][mat_nombre]
            c = color_map[mat_nombre]
            # PENELOPE ground truth
            ax.plot(datos['alturas_test'],
                    res['Y_true_2d'][:, j],
                    '-', color=c, linewidth=1.0, alpha=0.7,
                    label=f'{nombre_pub(mat_nombre)} (MC)')
            # Pipeline-RT
            ax.plot(datos['alturas_test'],
                    res['Y_pred_rt_2d'][:, j],
                    '--', color=c, linewidth=1.0, alpha=0.9)
            mapes_rt_this_E.append(
                res['metricas_por_energia'][j]['MAPE_RT_%'])
            mapes_pb_this_E.append(
                res['metricas_por_energia'][j]['MAPE_PB_%'])

        avg_mape_rt = np.mean(mapes_rt_this_E)
        avg_mape_pb = np.mean(mapes_pb_this_E)
        ax.set_xlabel('Height (cm)')
        ax.set_ylabel('FEPE')
        ax.set_title(f'{ek:.1f} keV - mean MAPE: '
                     f'RT={avg_mape_rt:.2f}%  PB={avg_mape_pb:.2f}%')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=6, ncol=2)

    for j_extra in range(n_E_test, len(axes_flat)):
        axes_flat[j_extra].set_visible(False)

    fig.suptitle(f'FEPE(h): Pipeline-RT vs PENELOPE - '
                 f'all materials - det. {det_label}\n'
                 f'(solid = MC, dashed = Pipeline-RT, '
                 f'unseen E and h)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / '09_RESUMEN_FEPE_vs_h_todos.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 10. MAPE por energía: barras agrupadas por material ----
    energias_labels = [f'{ek:.0f}' for ek in datos['energias_test']]
    n_mat = len(mat_list)
    n_ene = n_E_test

    fig, axes_mape = plt.subplots(2, 1, figsize=(12, 10))

    for i_corr, (tag, key_mape) in enumerate([('RT', 'MAPE_RT_%'),
                                                ('PB', 'MAPE_PB_%')]):
        ax = axes_mape[i_corr]
        w_total = 0.8
        w_bar = w_total / n_mat
        x_base = np.arange(n_ene)

        for i_m, mat_nombre in enumerate(mat_list):
            res = resultados['resultados_por_mat'][mat_nombre]
            vals = [m[key_mape] for m in res['metricas_por_energia']]
            offset = -w_total / 2 + w_bar * (i_m + 0.5)
            ax.bar(x_base + offset, vals, w_bar * 0.9,
                   color=color_map[mat_nombre], edgecolor='black',
                   linewidth=0.5, alpha=0.8, label=nombre_pub(mat_nombre))

        # Mean over materials
        for j in range(n_ene):
            vals_j = [resultados['resultados_por_mat'][m]
                      ['metricas_por_energia'][j][key_mape]
                      for m in mat_list]
            avg_j = np.mean(vals_j)
            ax.plot([x_base[j] - w_total/2 - 0.05,
                     x_base[j] + w_total/2 + 0.05],
                    [avg_j, avg_j], 'k-', linewidth=1.5, alpha=0.6)

        ax.set_xticks(x_base)
        ax.set_xticklabels(energias_labels)
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('MAPE (%)')
        # MAPE global medio
        mapes_glob = [resultados['resultados_por_mat'][m]
                      [f'metricas_{tag.lower()}']['MAPE_%']
                      for m in mat_list]
        ax.set_title(f'{tag} - Overall mean MAPE: {np.mean(mapes_glob):.2f}%')
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'MAPE per energy and material - det. {det_label} '
                 f'(unseen E and h)', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / '10_RESUMEN_MAPE_por_energia_todos.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 11. Heatmap material × energía (MAPE RT) ----
    mape_matrix_rt = np.zeros((n_mat, n_ene))
    mape_matrix_pb = np.zeros((n_mat, n_ene))
    for i_m, mat_nombre in enumerate(mat_list):
        res = resultados['resultados_por_mat'][mat_nombre]
        for j in range(n_ene):
            mape_matrix_rt[i_m, j] = res['metricas_por_energia'][j]['MAPE_RT_%']
            mape_matrix_pb[i_m, j] = res['metricas_por_energia'][j]['MAPE_PB_%']

    fig, axes_hm = plt.subplots(1, 2, figsize=(max(10, n_ene * 1.5), max(4, n_mat * 0.7)))
    for i_corr, (matrix, tag) in enumerate(
            [(mape_matrix_rt, 'RT'), (mape_matrix_pb, 'PB')]):
        ax = axes_hm[i_corr]
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                        vmin=0, vmax=max(matrix.max(), 1.0))
        ax.set_xticks(range(n_ene))
        ax.set_xticklabels(energias_labels, fontsize=9)
        ax.set_yticks(range(n_mat))
        ax.set_yticklabels([nombre_pub(m) for m in mat_list], fontsize=9)
        ax.set_xlabel('Energy (keV)')
        ax.set_title(f'MAPE (%) - {tag}')
        # Anotar valores
        for i_m in range(n_mat):
            for j in range(n_ene):
                ax.text(j, i_m, f'{matrix[i_m, j]:.1f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if matrix[i_m, j] > matrix.max() * 0.6
                        else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'MAPE heatmap: material × energy - det. {det_label}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / '11_RESUMEN_heatmap_MAPE.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 12. Relative residuals vs FEPE (diagnóstico) ----
    fig, axes_res = plt.subplots(1, 2, figsize=(14, 5.5))
    for i_corr, (tag, pred_key) in enumerate([('RT', 'Y_pred_mat_rt'),
                                                ('PB', 'Y_pred_mat_pb')]):
        ax = axes_res[i_corr]
        for mat_nombre in mat_list:
            res = resultados['resultados_por_mat'][mat_nombre]
            Y_r = res['Y_true_MC']
            Y_p = res[pred_key]
            residuo_rel = (Y_p - Y_r) / np.clip(np.abs(Y_r), 1e-12, None) * 100
            ax.scatter(Y_r, residuo_rel, s=6, alpha=0.4, edgecolors='none',
                       color=color_map[mat_nombre], label=nombre_pub(mat_nombre))
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax.axhline(5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axhline(-5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_xlabel('FEPE (PENELOPE)')
        ax.set_ylabel('Relative residual (%)')
        ax.set_title(f'{tag}')
        ax.legend(fontsize=7, markerscale=2, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Relative residuals vs FEPE - det. {det_label} '
                 f'(all materials, unseen E and h)', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / '12_RESUMEN_residuos_relativos.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Gráficas resumen (08-12) guardadas en {fig_dir}/')


# ============================================================================
# =================== EXPORTACIÓN JSON ======================================
# ============================================================================

def exportar_resultados_json(cfg, hist, datos, resultados):
    """
    Exporta un JSON exhaustivo con todos los datos necesarios para
    análisis posterior y generación de tablas/figuras del artículo.
    """
    fichero = cfg['fichero_modelo'].replace('.pt', '.json')
    n_epochs = len(hist['train_loss'])
    n_h_test = len(datos['idx_h_test'])
    n_E_test = len(datos['energias_test'])

    m_ann = resultados['metricas_ann']

    # --- Resumen por material (compacto) ---
    resumen_materiales = {}
    for mat_nombre, res in resultados['resultados_por_mat'].items():
        resumen_materiales[nombre_pub(mat_nombre)] = {
            'MAPE_RT_%': round(res['metricas_rt']['MAPE_%'], 3),
            'R2_RT': round(res['metricas_rt']['R2'], 6),
            'max_err_RT_%': round(res['metricas_rt']['max_err_%'], 3),
            'MAPE_PB_%': round(res['metricas_pb']['MAPE_%'], 3),
            'R2_PB': round(res['metricas_pb']['R2'], 6),
            'max_err_PB_%': round(res['metricas_pb']['max_err_%'], 3),
            'metricas_por_energia': res['metricas_por_energia'],
        }

    # --- Datos punto a punto por material (para generar tablas/figuras) ---
    datos_punto_a_punto = {}
    for mat_nombre, res in resultados['resultados_por_mat'].items():
        Y_true_2d = res['Y_true_2d']  # shape (n_h_test, n_E_test)
        Y_pred_rt_2d = res['Y_pred_rt_2d']
        Y_pred_pb_2d = res['Y_pred_pb_2d']
        f_a_rt_2d = res['f_a_rt'].reshape(n_h_test, n_E_test)
        f_a_pb_2d = res['f_a_pb'].reshape(n_h_test, n_E_test)
        f_a_MC_2d = res['f_a_MC'].reshape(n_h_test, n_E_test)

        # Arrays de alturas y energías test
        h_arr = datos['alturas_test']
        E_arr = datos['energias_test']

        registros = []
        for i_h in range(n_h_test):
            for j_E in range(n_E_test):
                fepe_true = float(Y_true_2d[i_h, j_E])
                fepe_rt = float(Y_pred_rt_2d[i_h, j_E])
                fepe_pb = float(Y_pred_pb_2d[i_h, j_E])
                registros.append({
                    'h_cm': round(float(h_arr[i_h]), 4),
                    'E_keV': round(float(E_arr[j_E]), 3),
                    'FEPE_MC': round(fepe_true, 6),
                    'FEPE_PINN_RT': round(fepe_rt, 6),
                    'FEPE_PINN_PB': round(fepe_pb, 6),
                    'err_rel_RT_%': round(abs(fepe_rt - fepe_true) /
                                         max(abs(fepe_true), 1e-12) * 100, 3),
                    'err_rel_PB_%': round(abs(fepe_pb - fepe_true) /
                                         max(abs(fepe_true), 1e-12) * 100, 3),
                    'f_a_RT': round(float(f_a_rt_2d[i_h, j_E]), 6),
                    'f_a_PB': round(float(f_a_pb_2d[i_h, j_E]), 6),
                    'f_a_MC': round(float(f_a_MC_2d[i_h, j_E]), 6),
                })

        datos_punto_a_punto[nombre_pub(mat_nombre)] = registros

    # --- ANN sola: métricas por energía sobre material de entrenamiento ---
    Y_pred_ann_2d = resultados['Y_pred_train_mat'].reshape(n_h_test, n_E_test)
    Y_true_ann_2d = resultados['Y_true_train_mat'].reshape(n_h_test, n_E_test)

    metricas_ann_por_energia = []
    for j, ek in enumerate(datos['energias_test']):
        col_t = Y_true_ann_2d[:, j]
        col_p = Y_pred_ann_2d[:, j]
        m_j = calcular_metricas(col_t, col_p)
        metricas_ann_por_energia.append({
            'energia_keV': round(float(ek), 3),
            'MAPE_%': round(m_j['MAPE_%'], 3),
            'R2': round(m_j['R2'], 6),
            'max_err_%': round(m_j['max_err_%'], 3),
        })

    # --- Métricas por altura por material ---
    metricas_por_altura = {}
    for mat_nombre, res in resultados['resultados_por_mat'].items():
        Y_true_2d = res['Y_true_2d']
        Y_pred_rt_2d = res['Y_pred_rt_2d']
        Y_pred_pb_2d = res['Y_pred_pb_2d']

        por_h = []
        for i_h in range(n_h_test):
            row_t = Y_true_2d[i_h, :]
            row_rt = Y_pred_rt_2d[i_h, :]
            row_pb = Y_pred_pb_2d[i_h, :]
            m_h_rt = calcular_metricas(row_t, row_rt)
            m_h_pb = calcular_metricas(row_t, row_pb)
            por_h.append({
                'h_cm': round(float(datos['alturas_test'][i_h]), 4),
                'MAPE_RT_%': round(m_h_rt['MAPE_%'], 3),
                'MAPE_PB_%': round(m_h_pb['MAPE_%'], 3),
                'R2_RT': round(m_h_rt['R2'], 6),
                'R2_PB': round(m_h_pb['R2'], 6),
            })
        metricas_por_altura[nombre_pub(mat_nombre)] = por_h

    # --- Métricas pooled (all materials combinados) ---
    metricas_pooled = {}
    if 'metricas_pooled_rt' in resultados:
        m_p_rt = resultados['metricas_pooled_rt']
        m_p_pb = resultados['metricas_pooled_pb']
        metricas_pooled = {
            'n_materiales': len(resultados['resultados_por_mat']),
            'n_puntos_total': int(len(resultados['resultados_por_mat'])
                                  * n_h_test * n_E_test),
            'MAPE_RT_%': round(m_p_rt['MAPE_%'], 3),
            'R2_RT': round(m_p_rt['R2'], 6),
            'max_err_RT_%': round(m_p_rt['max_err_%'], 3),
            'MAPE_PB_%': round(m_p_pb['MAPE_%'], 3),
            'R2_PB': round(m_p_pb['R2'], 6),
            'max_err_PB_%': round(m_p_pb['max_err_%'], 3),
        }

    # --- Resumen estadístico cross-material ---
    if len(resultados['resultados_por_mat']) > 0:
        mapes_rt_all = [res['metricas_rt']['MAPE_%']
                        for res in resultados['resultados_por_mat'].values()]
        mapes_pb_all = [res['metricas_pb']['MAPE_%']
                        for res in resultados['resultados_por_mat'].values()]
        resumen_estadistico = {
            'MAPE_RT_media_%': round(float(np.mean(mapes_rt_all)), 3),
            'MAPE_RT_std_%': round(float(np.std(mapes_rt_all)), 3),
            'MAPE_RT_min_%': round(float(np.min(mapes_rt_all)), 3),
            'MAPE_RT_max_%': round(float(np.max(mapes_rt_all)), 3),
            'MAPE_PB_media_%': round(float(np.mean(mapes_pb_all)), 3),
            'MAPE_PB_std_%': round(float(np.std(mapes_pb_all)), 3),
            'MAPE_PB_min_%': round(float(np.min(mapes_pb_all)), 3),
            'MAPE_PB_max_%': round(float(np.max(mapes_pb_all)), 3),
        }
    else:
        resumen_estadistico = {}

    salida = {
        'version': 'PINN_POZO_multimat_v2',
        'detector': 'Pozo HPGe (GIRMA) con vial cilíndrico',
        'proposito': 'PINN Nivel 2: ANN trained on material de calibración, '
                     'corrección de autoatenuación por ray-tracing y parallel beam. '
                     'Test multi-material en alturas y energías no vistas.',
        'correcciones_fa': ['ray-tracing geométrico (caminos oblicuos, geometría pozo)',
                            'parallel beam (haz paralelo)'],
        'raytracing_config': {'N_source': 8000, 'N_det': 400},
        'geometria_pozo': {
            'radio_pozo_cristal_cm': 1.336516,
            'profundidad_pozo_cm': 3.473473,
            'radio_vial_interno_cm': 0.575,
        },
        'material_train': nombre_pub(cfg['material_train']),
        'alturas_train_cm': cfg['alturas_train_cm'],
        'alturas_test_cm': [round(float(h), 4)
                            for h in datos['alturas_test']],
        'energias_train_keV': [round(float(e), 3)
                               for e in datos['energias_train']],
        'energias_test_keV': [round(float(e), 3)
                              for e in datos['energias_test']],
        'n_alturas_test': int(n_h_test),
        'n_energias_test': int(n_E_test),
        'composiciones': {nombre_pub(k): v['composicion']
                          for k, v in MATERIALES.items()},
        'densidades': {nombre_pub(k): v['densidad']
                       for k, v in MATERIALES.items()},

        # ANN sola
        'metricas_ANN_sola': {
            'material': nombre_pub(cfg['material_train']),
            'MAPE_%': round(m_ann['MAPE_%'], 3),
            'R2': round(m_ann['R2'], 6),
            'max_err_%': round(m_ann['max_err_%'], 3),
            'nota': 'Evaluada solo en alturas y energías NO vistas',
            'por_energia': metricas_ann_por_energia,
        },

        # Pipeline por material (resumen)
        'metricas_pipeline_por_material': resumen_materiales,

        # Métricas pooled (all materials juntos)
        'metricas_pooled_todos_materiales': metricas_pooled,

        # Resumen estadístico cross-material
        'resumen_estadistico_cross_material': resumen_estadistico,

        # Métricas por altura por material
        'metricas_por_altura_por_material': metricas_por_altura,

        # Datos punto a punto (para tablas y figuras del artículo)
        'datos_punto_a_punto': datos_punto_a_punto,

        # Entrenamiento
        'entrenamiento': {
            'epochs': n_epochs,
            'mejor_val_loss': round(float(min(hist['val_loss'])), 6),
            'arquitectura': cfg['capas_ocultas'],
            'activacion': cfg['activacion'],
            'normalizacion_X': cfg['normalizacion_X'],
            'normalizacion_Y': cfg['normalizacion_Y'],
            'log_E': cfg.get('log_E', False),
            'learning_rate': cfg['learning_rate'],
            'weight_decay': cfg['weight_decay'],
            'batch_size': cfg['batch_size'],
        },
    }
    with open(fichero, 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f'\nResultados exportados a {fichero}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    cfg = CONFIG

    print('=' * 60)
    print('  PINN Nivel 2: ANN + autoatenuación RT/PB - well det.')
    print('  Versión multi-material con test en puntos no vistos')
    print('=' * 60)
    print(f'\n  Fichero train: {cfg["fichero_train"]}  ({nombre_pub(cfg["material_train"])})')
    print(f'  Buscando materiales de test: {cfg["patron_excels"]}')

    xcom = TablaXCOM(cfg['fichero_xcom'])
    print(f'\n  Tabla XCOM cargada: {len(xcom.energias_keV)} energías, '
          f'{len(xcom.elementos)} elementos')

    datos = preparar_datos(cfg, xcom)
    modelo, hist = entrenar(cfg, datos)
    resultados = evaluar(cfg, modelo, datos)
    generar_graficas(cfg, hist, datos, resultados, xcom)
    generar_graficas_resumen(cfg, datos, resultados)
    exportar_resultados_json(cfg, hist, datos, resultados)

    if cfg['guardar_modelo']:
        torch.save({
            'estado_modelo': modelo.state_dict(),
            'config': cfg,
            'materiales': MATERIALES,
            'norm_X_params': datos['norm_X'].params,
            'norm_X_metodo': datos['norm_X'].metodo,
            'norm_Y_params': datos['norm_Y'].params,
            'norm_Y_metodo': datos['norm_Y'].metodo,
        }, cfg['fichero_modelo'])
        print(f'Modelo guardado en {cfg["fichero_modelo"]}')

    print('\n¡Terminado!')
