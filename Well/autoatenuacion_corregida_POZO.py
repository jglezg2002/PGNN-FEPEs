#!/usr/bin/env python3
"""
autoatenuacion_corregida_POZO.py

Ray-tracing geométrico para calcular F_corr en el detector pozo (GIRMA)
con vial cilíndrico.  F_corr = sum(w_i * exp(-mu*rho*l_i)) / sum(w_i).

Pesos: 'uniforme' (w=1), 'cos' (w=cos th), 'solido' (w=cos th / d^2).
cos th se toma respecto a la normal local de cada superficie del well
(radial en la pared cilíndrica, axial en el disco inferior).

Geometría sacada de CalibracionPOZO_BarridoAlturas.py.
J. G. Guerra
"""

import numpy as np
import time

PESOS_VALIDOS = ('uniforme', 'cos', 'solido')

# -- Geometría del detector pozo GIRMA --

DIM = [7.4, 3.284199, 1.336516, 3.473473, 0.043489, 1.174834, 0.207584]

_longitudcristal        = DIM[0]
_radiocristal           = DIM[1]
_radiopozocristal       = DIM[2]
_profpozocristal        = DIM[3]
_espesorcapamuerta      = DIM[4]
_distanciacristalcubierta = DIM[5]
_espesorcubierta        = DIM[6]
_profpozocubierta       = 4.0
_espesorvaso            = 0.1
_radiointvaso           = 1.15 / 2
_longitudvaso           = 5.3

# Cotas z
_z_base         = 0.5
_z_crystal_top  = _z_base + _longitudcristal
_z_well_bottom  = _z_crystal_top - _profpozocristal
_z_gap_top      = _z_crystal_top + _distanciacristalcubierta
_z_endcap_top   = _z_gap_top + _espesorcubierta
_z_vial_start   = _z_endcap_top - _profpozocubierta + _espesorvaso
_z_sample_base  = _z_vial_start

_r_well    = _radiopozocristal
_r_sample  = _radiointvaso

# Áreas de la superficie interior del well (para repartir los puntos)
_h_well         = _z_crystal_top - _z_well_bottom
_area_cylinder  = 2 * np.pi * _r_well * _h_well
_area_disc      = np.pi * _r_well**2
_area_total     = _area_cylinder + _area_disc
_frac_cylinder  = _area_cylinder / _area_total


# -- Ray-tracing vectorizado --

def precalcular_rayos(h_muestra, N_source=5000, N_det=300, semilla=42,
                      peso='solido'):
    """Precalcula caminos ópticos y pesos para la geometría del well."""
    if peso not in PESOS_VALIDOS:
        raise ValueError(f"peso debe ser uno de {PESOS_VALIDOS}, got '{peso}'")

    rng = np.random.RandomState(semilla)

    # Puntos fuente uniformes en el cilindro de muestra
    r_s = _r_sample * np.sqrt(rng.uniform(0, 1, N_source))
    z_s_local = rng.uniform(0, h_muestra, N_source)
    z_s_abs = _z_sample_base + z_s_local

    # Puntos detector en la superficie interior del well
    N_cyl = int(round(N_det * _frac_cylinder))
    N_disc = N_det - N_cyl

    # Pared cilíndrica
    phi_cyl = rng.uniform(0, 2 * np.pi, N_cyl)
    z_cyl = rng.uniform(_z_well_bottom, _z_crystal_top, N_cyl)
    x_cyl = _r_well * np.cos(phi_cyl)
    y_cyl = _r_well * np.sin(phi_cyl)

    # Disco inferior del well
    r_disc = _r_well * np.sqrt(rng.uniform(0, 1, N_disc))
    phi_disc = rng.uniform(0, 2 * np.pi, N_disc)
    x_disc = r_disc * np.cos(phi_disc)
    y_disc = r_disc * np.sin(phi_disc)
    z_disc = np.full(N_disc, _z_well_bottom)

    x_d = np.concatenate([x_cyl, x_disc])
    y_d = np.concatenate([y_cyl, y_disc])
    z_d = np.concatenate([z_cyl, z_disc])

    # Normales locales (apuntan hacia dentro del well)
    nx_d = np.concatenate([-np.cos(phi_cyl), np.zeros(N_disc)])
    ny_d = np.concatenate([-np.sin(phi_cyl), np.zeros(N_disc)])
    nz_d = np.concatenate([np.zeros(N_cyl),  np.ones(N_disc)])

    Nd = len(x_d)

    # Bloques para no reventar la RAM
    BLOCK = 1000
    all_paths = []
    all_weights = []

    for i0 in range(0, N_source, BLOCK):
        i1 = min(i0 + BLOCK, N_source)
        Ns_b = i1 - i0

        rs = r_s[i0:i1]
        zs_loc = z_s_local[i0:i1]
        zs_abs = z_s_abs[i0:i1]

        # Vectores fuente -> detector
        dx = x_d[np.newaxis, :] - rs[:, np.newaxis]
        dy = np.broadcast_to(y_d[np.newaxis, :], (Ns_b, Nd))
        dz = z_d[np.newaxis, :] - zs_abs[:, np.newaxis]

        d_total = np.sqrt(dx**2 + dy**2 + dz**2)

        if peso == 'uniforme':
            w = np.ones_like(d_total)
        elif peso == 'cos':
            dot_dn = (dx * nx_d[np.newaxis, :] +
                      dy * ny_d[np.newaxis, :] +
                      dz * nz_d[np.newaxis, :])
            cos_theta_local = np.abs(dot_dn) / d_total
            w = cos_theta_local
        elif peso == 'solido':
            dot_dn = (dx * nx_d[np.newaxis, :] +
                      dy * ny_d[np.newaxis, :] +
                      dz * nz_d[np.newaxis, :])
            cos_theta_local = np.abs(dot_dn) / d_total
            w = cos_theta_local / d_total**2

        zs_2d = zs_loc[:, np.newaxis]
        rs_2d = rs[:, np.newaxis]
        h = h_muestra

        # Salida por fondo (z_local = 0)
        t_bot = np.where(dz < -1e-15, -zs_2d / dz, 1e30)
        x_at_bot = rs_2d + t_bot * dx
        y_at_bot = t_bot * dy
        r_at_bot = np.sqrt(x_at_bot**2 + y_at_bot**2)
        t_bot = np.where((t_bot > 1e-10) & (r_at_bot <= _r_sample + 1e-6),
                         t_bot, 1e30)

        # Salida por techo (z_local = h)
        t_top = np.where(dz > 1e-15, (h - zs_2d) / dz, 1e30)
        x_at_top = rs_2d + t_top * dx
        y_at_top = t_top * dy
        r_at_top = np.sqrt(x_at_top**2 + y_at_top**2)
        t_top = np.where((t_top > 1e-10) & (r_at_top <= _r_sample + 1e-6),
                         t_top, 1e30)

        # Salida por pared lateral (r = r_sample)
        A_q = dx**2 + dy**2
        B_q = 2 * rs_2d * dx
        C_q = rs_2d**2 - _r_sample**2

        disc = B_q**2 - 4 * A_q * C_q
        disc_ok = (disc >= 0) & (A_q > 1e-15)
        sqrt_disc = np.sqrt(np.maximum(disc, 0))

        t_side = np.where(disc_ok, (-B_q + sqrt_disc) / (2 * A_q), 1e30)
        z_at_side = zs_2d + t_side * dz
        t_side = np.where(
            (t_side > 1e-10) & (z_at_side >= -1e-6) & (z_at_side <= h + 1e-6),
            t_side, 1e30
        )

        # Camino = primer cruce con la pared
        t_exit = np.minimum(np.minimum(t_bot, t_top), t_side)
        path = t_exit * d_total
        path = np.maximum(path, 0)
        path = np.where(t_exit < 1e20, path, 0)

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
    """Fórmula clásica de haz paralelo: F = [1-exp(-μρh)] / (μρh)"""
    x = np.atleast_1d(np.float64(mu_rho)) * rho * h
    return np.where(x > 1e-10, (1 - np.exp(-x)) / x, 1.0 - x / 2)


class CacheRayos:
    """Almacena caminos precalculados para no recalcularlos cada vez."""
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
    print('=' * 70)
    print('  F paralelo vs F corregido — ray-tracing detector POZO')
    print('  Comparación de esquemas de peso')
    print('=' * 70)

    print(f'\n  Geometría del well:')
    print(f'    r_well = {_r_well:.4f} cm')
    print(f'    h_well = {_h_well:.4f} cm '
          f'(z: {_z_well_bottom:.3f} -> {_z_crystal_top:.3f})')
    print(f'    r_sample = {_r_sample:.4f} cm')
    print(f'    z_sample_base = {_z_sample_base:.4f} cm')
    print(f'    Área cilindro: {_area_cylinder:.2f} cm² '
          f'({_frac_cylinder*100:.0f}%)')
    print(f'    Área disco:    {_area_disc:.2f} cm² '
          f'({(1-_frac_cylinder)*100:.0f}%)')

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
    print(f'\n  Cache 51 alturas ({Ns}×{Nd}, uniforme):')
    cache = CacheRayos(Ns, Nd, peso='uniforme')
    t0 = time.time()
    for h_i in np.arange(0.1, 5.2, 0.1):
        cache.get(round(h_i, 1))
    print(f'  Total: {time.time()-t0:.1f} s')
