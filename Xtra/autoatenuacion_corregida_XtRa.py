#!/usr/bin/env python3
"""
autoatenuacion_corregida_XtRa.py

Ray-tracing geométrico para calcular F_corr en el detector XtRa plano
con bote troncocónico (tipo IAEA-448).
F_corr = sum(w_i * exp(-mu*rho*l_i)) / sum(w_i).

Pesos: 'uniforme' (w=1), 'cos' (w=cos th), 'solido' (w=cos th / d^2).

Geometría sacada de CalibracionXtRa_BarridoAlturas.py.
J. G. Guerra
"""

import numpy as np
import time

PESOS_VALIDOS = ('uniforme', 'cos', 'solido')

# -- Geometría XtRa + bote troncocónico --

DIM = [0.172689, 0.529202, 0.091671, 0.875876, 2.452944, 0.290189]
_radioger          = 2.976
_espesorcapamuerta = DIM[2]
_cotasuperior      = 0.5 - DIM[1]
_r_ge_active       = _radioger - _espesorcapamuerta
_cotainferiorventana = 0.5
_espesorberilio      = 0.05
_espesorinfvaso      = 0.1
_radiointvasoinf = 4.65 / 2
_radiointvasosup = 5.46 / 2
_longitudvaso    = 7.37
_z_bote_start  = _cotainferiorventana + _espesorberilio + 0.1
_z_sample_base = _z_bote_start + _espesorinfvaso
_dr_dz = (_radiointvasosup - _radiointvasoinf) / _longitudvaso
_z_det = _cotasuperior
_r_det = _r_ge_active


# -- Ray-tracing vectorizado --

def precalcular_rayos(h_muestra, N_source=5000, N_det=300, semilla=42,
                      peso='solido'):
    """Precalcula caminos ópticos y pesos para la geometría XtRa."""
    if peso not in PESOS_VALIDOS:
        raise ValueError(f"peso debe ser uno de {PESOS_VALIDOS}, got '{peso}'")

    rng = np.random.RandomState(semilla)

    # Puntos fuente uniformes en el tronco de cono
    r_max = _radiointvasoinf + _dr_dz * h_muestra
    src_ok = np.zeros((0, 2))
    while len(src_ok) < N_source:
        n_gen = int((N_source - len(src_ok)) * 2.0) + 500
        z_c = rng.uniform(0, h_muestra, n_gen)
        r_c = r_max * np.sqrt(rng.uniform(0, 1, n_gen))
        r_wall = _radiointvasoinf + _dr_dz * z_c
        ok = r_c <= r_wall
        new = np.column_stack([r_c[ok], z_c[ok]])
        src_ok = np.vstack([src_ok, new]) if len(src_ok) > 0 else new
    src_ok = src_ok[:N_source]

    # Puntos detector (disco plano de Ge activo)
    r_d = _r_det * np.sqrt(rng.uniform(0, 1, N_det))
    phi_d = rng.uniform(0, 2 * np.pi, N_det)
    x_d = r_d * np.cos(phi_d)
    y_d = r_d * np.sin(phi_d)

    BLOCK = 1000
    all_paths = []
    all_weights = []

    for i0 in range(0, N_source, BLOCK):
        i1 = min(i0 + BLOCK, N_source)
        Ns_b = i1 - i0
        Nd = N_det

        r_s = src_ok[i0:i1, 0]
        z_s_local = src_ok[i0:i1, 1]
        z_s_abs = _z_sample_base + z_s_local

        dx = x_d[np.newaxis, :] - r_s[:, np.newaxis]
        dy_arr = np.broadcast_to(y_d[np.newaxis, :], (Ns_b, Nd))
        dz = _z_det - z_s_abs[:, np.newaxis]
        d_total = np.sqrt(dx**2 + dy_arr**2 + dz**2)

        cos_theta = np.abs(dz) / d_total

        if peso == 'uniforme':
            w = np.ones_like(d_total)
        elif peso == 'cos':
            w = cos_theta
        elif peso == 'solido':
            w = cos_theta / d_total**2

        z_s_2d = z_s_local[:, np.newaxis]
        r_s_2d = r_s[:, np.newaxis]

        # Cruce con el fondo
        t_bot = -z_s_2d / dz
        x_ex = r_s_2d + t_bot * dx
        y_ex = t_bot * dy_arr
        r_ex = np.sqrt(x_ex**2 + y_ex**2)
        exits_bot = (r_ex <= _radiointvasoinf + 0.01) & (t_bot > 0)
        path_bot = t_bot * d_total

        # Cruce con la pared cónica
        slope = _dr_dz
        Rz0 = _radiointvasoinf + slope * z_s_2d
        A = dx**2 + dy_arr**2 - (slope * dz)**2
        B = 2 * r_s_2d * dx - 2 * Rz0 * slope * dz
        C = r_s_2d**2 - Rz0**2
        disc = B**2 - 4 * A * C
        disc_ok = (disc >= 0) & (np.abs(A) > 1e-15)
        sqrt_d = np.sqrt(np.maximum(disc, 0))

        s1 = np.where(disc_ok, (-B + sqrt_d) / (2 * A), 1e30)
        s2 = np.where(disc_ok, (-B - sqrt_d) / (2 * A), 1e30)
        z1 = z_s_2d + s1 * dz
        z2 = z_s_2d + s2 * dz
        v1 = (s1 > 1e-10) & (z1 >= -0.01) & (z1 <= h_muestra + 0.01)
        v2 = (s2 > 1e-10) & (z2 >= -0.01) & (z2 <= h_muestra + 0.01)

        s_cone = np.full_like(s1, 1e30)
        s_cone = np.where(v1, np.minimum(s_cone, s1), s_cone)
        s_cone = np.where(v2, np.minimum(s_cone, s2), s_cone)
        path_cone = s_cone * d_total

        path_bot_valid = np.where(exits_bot, path_bot, 1e30)
        path = np.minimum(path_bot_valid, path_cone)
        path = np.maximum(path, 0)

        all_paths.append(path.ravel())
        all_weights.append(w.ravel())

    return np.concatenate(all_paths), np.concatenate(all_weights)


def F_corregido(mu_rho, rho, path_lengths, weights):
    """F_corr = Σ(w·exp(-μρℓ)) / Σ(w)"""
    mu_rho = np.atleast_1d(np.float64(mu_rho))
    w_sum = weights.sum()
    if w_sum < 1e-30:
        return np.ones(len(mu_rho))

    F = np.zeros(len(mu_rho))
    for j, mu in enumerate(mu_rho):
        exp_vals = np.exp(np.clip(-mu * rho * path_lengths, -500, 0))
        F[j] = np.dot(weights, exp_vals) / w_sum

    return F if len(F) > 1 else float(F[0])


def F_parallel_beam(mu_rho, rho, h):
    x = np.atleast_1d(np.float64(mu_rho)) * rho * h
    return np.where(x > 1e-10, (1 - np.exp(-x)) / x, 1.0 - x / 2)


class CacheRayos:
    def __init__(self, N_source=8000, N_det=400, peso='solido'):
        self.N_source = N_source
        self.N_det = N_det
        self.peso = peso
        self._cache = {}

    def get(self, h_cm, decimales=3):
        key = round(h_cm, decimales)
        if key not in self._cache:
            self._cache[key] = precalcular_rayos(
                key, self.N_source, self.N_det, peso=self.peso)
        return self._cache[key]


def correccion_material_raytracing(xcom, comp_muestra, rho_muestra,
                                   comp_patron, rho_patron,
                                   E_keV, h_cm, cache=None):
    """f_a = F_corr(muestra) / F_corr(patrón)"""
    E_keV = np.atleast_1d(np.float64(E_keV))
    h_cm = np.atleast_1d(np.float64(h_cm))
    if cache is None:
        cache = CacheRayos()

    if len(h_cm) == 1 and len(E_keV) > 1:
        paths, wts = cache.get(h_cm[0])
        mu_m = xcom.mu_rho_mezcla(comp_muestra, E_keV)
        mu_p = xcom.mu_rho_mezcla(comp_patron, E_keV)
        F_m = F_corregido(mu_m, rho_muestra, paths, wts)
        F_p = F_corregido(mu_p, rho_patron, paths, wts)
        return np.where(F_p > 1e-30, F_m / F_p, 1.0)

    results = np.zeros(len(h_cm))
    for i in range(len(h_cm)):
        E_i = E_keV[i] if len(E_keV) > 1 else E_keV[0]
        paths, wts = cache.get(h_cm[i])
        mu_m = xcom.mu_rho_mezcla(comp_muestra, np.array([E_i]))
        mu_p = xcom.mu_rho_mezcla(comp_patron, np.array([E_i]))
        F_m = F_corregido(mu_m, rho_muestra, paths, wts)
        F_p = F_corregido(mu_p, rho_patron, paths, wts)
        results[i] = F_m / F_p if F_p > 1e-30 else 1.0
    return results


# ---- Test rápido ----

if __name__ == '__main__':
    print('='*70)
    print('  F paralelo vs F corregido — ray-tracing XtRa')
    print('  Comparación de esquemas de peso')
    print('='*70)

    h = 2.5
    rho = 1.4277
    mus = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    F_pb = F_parallel_beam(mus, rho, h)

    Ns, Nd = 5000, 300

    for esquema in PESOS_VALIDOS:
        t0 = time.time()
        p, w = precalcular_rayos(h, Ns, Nd, peso=esquema)
        dt = time.time() - t0

        F_rt = F_corregido(mus, rho, p, w)

        print(f'\n  ── peso = {esquema!r}  ({dt:.1f} s) ──')
        print(f'  Caminos: min={p.min():.3f}, mean={p.mean():.3f}, '
              f'max={p.max():.3f}')
        print(f'  k_eff = {p.mean()/(h/2):.4f}')
        print(f'\n  {"μ/ρ":>6} {"F_PB":>10} {"F_RT":>10} '
              f'{"Δ(RT-PB)%":>10}')
        print('  ' + '-'*40)
        for j, mu in enumerate(mus):
            d = (F_rt[j] - F_pb[j]) / F_pb[j] * 100
            print(f'  {mu:6.1f} {F_pb[j]:10.6f} {F_rt[j]:10.6f} '
                  f'{d:+10.2f}')

    # Ratio f_a para un material ficticio (mu_m / mu_p = 2)
    print('\n' + '='*70)
    print('  Ratio f_a = F_corr(μ_m) / F_corr(μ_p)  con μ_m/μ_p = 2')
    print('  (μ_p = 0.5 → μ_m = 1.0)')
    print('='*70)

    mu_p_test = 0.5
    mu_m_test = 1.0

    f_pb = (F_parallel_beam(mu_m_test, rho, h) /
            F_parallel_beam(mu_p_test, rho, h)).item()
    print(f'\n  f_a (PB)       = {f_pb:.6f}')

    for esquema in PESOS_VALIDOS:
        p, w = precalcular_rayos(h, Ns, Nd, peso=esquema)
        F_m = float(F_corregido(mu_m_test, rho, p, w))
        F_p = float(F_corregido(mu_p_test, rho, p, w))
        f_rt = F_m / F_p
        delta = (f_rt - f_pb) / f_pb * 100
        print(f'  f_a ({esquema:<9s}) = {f_rt:.6f}   Δ vs PB: {delta:+.2f}%')

    # Timing de la cache
    print(f'\n  Cache 72 alturas ({Ns}×{Nd}, uniforme):')
    cache = CacheRayos(Ns, Nd, peso='uniforme')
    t0 = time.time()
    for h_i in np.arange(0.1, 7.3, 0.1):
        cache.get(round(h_i, 1))
    print(f'  Total: {time.time()-t0:.1f} s')
