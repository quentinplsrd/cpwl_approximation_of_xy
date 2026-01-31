from typing import Tuple, List, Hashable
import pandas as pd
import numpy as np

def create_rough_zones(
    q_min_df: pd.DataFrame,
    q_max_df: pd.DataFrame,
    row_label: Hashable | None = None,
    upper_cap: float = 1e4
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build "intermediate" zones using a different rule:
      - zone_min[k] = sorted_unique(q_max)[k]
      - zone_max[k] = sorted_unique(q_min)[k+1] if available else upper_cap

    Intuition:
      Given ascending rough-mins M = [m0, m1, m2, ...] and rough-maxs X = [x0, x1, x2, ...],
      define zones as:
        Z1: [x0, m1)
        Z2: [x1, m2)
        ...
        ZK: [x_{K-1}, upper_cap)
      where K = len(X). This matches the example:
        q_min: [0, 4000, 6800]
        q_max: [2400, 4800, 7200]
        -> zone_mins  = [2400, 4800, 7200]
           zone_maxs  = [4000, 6800, 1e6]

    Parameters
    ----------
    q_min_df : pd.DataFrame
        DataFrame of shape (1, N) or with multiple rows; contains rough-min values (may contain NaN).
    q_max_df : pd.DataFrame
        DataFrame of shape (1, N) or with multiple rows; contains rough-max values (may contain NaN).
    row_label : hashable, optional
        Index label specifying which row (plant) to process; if None, uses the first row.
    upper_cap : float, optional
        Upper bound for the last zone (default = 1e6).

    Returns
    -------
    Q_rough_zone_min_zones : pd.DataFrame
        Indexed by zone name ['Z1','Z2',…], single column = [plant_name], values = zone minima.
    Q_rough_zone_max_zones : pd.DataFrame
        Indexed by zone name ['Z1','Z2',…], single column = [plant_name], values = zone maxima.
    rough_zones : list of str
        ['Z1','Z2',…]
    """
    # 1) Select the plant (row) to process
    if row_label is None:
        row_label = q_min_df.index[0]
    plant_name = row_label

    s_min = q_min_df.loc[row_label]
    s_max = q_max_df.loc[row_label]

    # 2) Extract numeric values, drop NaNs, sort ascending, and deduplicate
    mins = np.unique(s_min.dropna().astype(float).to_numpy())
    maxs = np.unique(s_max.dropna().astype(float).to_numpy())

    # 3) Basic validations
    if mins.size == 0 and maxs.size == 0:
        raise ValueError(f"No valid rough-min or rough-max boundaries for plant {plant_name}")
    if maxs.size == 0:
        raise ValueError(f"No valid rough-max boundaries for plant {plant_name}; cannot define zones")

    # 4) Build zones:
    #    zone_min[i] = maxs[i]
    #    zone_max[i] = mins[i+1] if i+1 exists else upper_cap
    zone_count = int(maxs.size)
    zone_mins = np.empty(zone_count, dtype=float)
    zone_maxs = np.empty(zone_count, dtype=float)

    for i in range(zone_count):
        zone_mins[i] = maxs[i]
        zone_maxs[i] = mins[i + 1] if (i + 1) < mins.size else float(upper_cap)

    # 5) Optional sanity check: enforce strictly increasing min < max; drop invalid zones
    valid_mask = zone_mins < zone_maxs
    if not np.all(valid_mask):
        zone_mins = zone_mins[valid_mask]
        zone_maxs = zone_maxs[valid_mask]
        zone_count = int(zone_mins.size)
        if zone_count == 0:
            raise ValueError(
                f"All computed zones are invalid (min >= max) for plant {plant_name}. "
                "Check your rough-min/rough-max inputs."
            )

    # 6) Name the zones and assemble output DataFrames
    zone_names = [f"Z{i + 1}" for i in range(zone_count)]
    Q_rough_zone_min_zones = pd.DataFrame({plant_name: zone_mins}, index=zone_names)
    Q_rough_zone_max_zones = pd.DataFrame({plant_name: zone_maxs}, index=zone_names)

    return Q_rough_zone_min_zones, Q_rough_zone_max_zones, zone_names