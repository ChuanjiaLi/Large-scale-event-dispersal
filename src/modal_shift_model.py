"""Reference implementation of the modal-shift scenarios (Methods, "Modal-shift scenarios"; Note S5.1).
"""
import numpy as np
from scipy import ndimage

PUBLIC = ('SUBWAY', 'BUS')      # metro, bus
PRIVATE = ('DRV', 'BYC')        # car, bicycle
TAU = np.linspace(-0.5, 0.5, 20)


def shifted_shares(shares, tau, public=PUBLIC, private=PRIVATE):
    """Mode shares after moving |tau| of all trips between the public and private groups."""
    if tau == 0:
        return dict(shares)
    donor, receiver = (private, public) if tau > 0 else (public, private)
    donor = [m for m in donor if m in shares]
    receiver = [m for m in receiver if m in shares]
    s_d = sum(shares[m] for m in donor)
    s_r = sum(shares[m] for m in receiver)
    delta = min(abs(tau), s_d)
    new = dict(shares)
    for m in donor:
        new[m] -= delta * shares[m] / s_d
    for m in receiver:
        new[m] += delta * (shares[m] / s_r if s_r > 0 else 1 / len(receiver))
    return new


def scenario_field(mode_grids, tau):
    """D'(tau) = M * sum_m s'_m P_m for one record."""
    totals = {m: float(g.sum()) for m, g in mode_grids.items()}
    M = sum(totals.values())
    shares = {m: t / M for m, t in totals.items()}
    patterns = {m: (g / totals[m] if totals[m] > 0 else np.zeros_like(g)) for m, g in mode_grids.items()}
    new = shifted_shares(shares, tau)
    return M * sum(new[m] * patterns[m] for m in mode_grids)


def peak_intensity(field):
    """Largest cell value of the scenario field."""
    return float(np.max(field))


def peak_weighted_radius(field, grid_x, grid_y):
    """Spread of the busiest destinations: weights D^4 on positive cells, around the flow-weighted centroid
    taken at the nearest grid cell; coordinate differences in degrees x 111 km. Returns 0 when fewer than five
    cells are positive."""
    pos = np.maximum(field, 0)
    if pos.sum() < 1e-6 or (pos > 0).sum() < 5:
        return 0.0
    r, c = ndimage.center_of_mass(pos)
    cx, cy = grid_x[int(round(r)), int(round(c))], grid_y[int(round(r)), int(round(c))]
    w = pos ** 4
    dist2_km = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) * 111 ** 2
    return float(np.sqrt((dist2_km * w).sum() / w.sum()))


def scenario_metrics(mode_grids, grid_x, grid_y, taus=TAU):
    """Peak intensity and peak-weighted radius for every transfer share (one row per tau)."""
    rows = []
    for t in taus:
        f = scenario_field(mode_grids, t)
        rows.append((float(t), peak_intensity(f), peak_weighted_radius(f, grid_x, grid_y)))
    return rows
