import numpy as np
import numba

"""
Thermodynamic routines for cloud tracking.

Temperature calculation uses the UCLA-LES saturation adjustment approach 
(Stevens et al. 2005) with Newton-Raphson iteration:
    T = theta_l * Pi + L_v * r_l / c_pd

where Pi = (p/p_0)^(R_d/c_pd) is the Exner function.

For saturated points (where diagnosed r_l > 0), we iterate to find T such that
r_l = r_condensable - r_s(T, p), using the Clausius-Clapeyron equation for dr_s/dT.

This replaces the Betts (1973) approximation which has a singularity at 
q_l = c_pd/L_v ≈ 0.4 g/kg - a value easily reached in cumulus clouds.

References:
- Stevens, B., et al. (2005). Evaluation of large-eddy simulations via observations 
  of nocturnal marine stratocumulus. Mon. Wea. Rev., 133, 1443-1462.
- Khairoutdinov, M. F., & Randall, D. A. (2003). Cloud resolving modeling of the 
  ARM summer 1997 IOP: Model formulation, results, uncertainties, and sensitivities.
  J. Atmos. Sci., 60, 607-625.

Note: RICO LES data water species (l, q) are in kg/kg despite the NetCDF metadata 
labeling them as g/kg. No unit conversion is needed in this code.
"""

# Module-level constants for use across functions
R_D = 287.04        # Gas constant for dry air [J/kg/K]
R_V = 461.5         # Gas constant for water vapor [J/kg/K]
C_PD = 1005.0       # Specific heat of dry air at constant pressure [J/kg/K]
C_PV = 1850.0       # Specific heat of water vapor at constant pressure [J/kg/K]
L_V = 2.5e6         # Latent heat of vaporization [J/kg]
P_0 = 100000.0      # Reference pressure [Pa]
EPSILON = 0.622     # Ratio of gas constants R_d/R_v (M_v/M_d)
G = 9.81            # Gravitational acceleration [m/s^2]


def compute_buoyancy_3d(theta_l, q_t, q_l, p, r=None):
    """Compute buoyancy (m/s^2) relative to horizontal domain mean θ_v at each level.

    Uses the LES-diagnosed q_l directly, so no saturation adjustment is needed.
    Equivalent to the density-based buoyancy in _density_with_loading, but operates
    on potential temperature rather than absolute temperature.

    Water loading includes cloud liquid and (optionally) rain, consistent with
    _density_with_loading.
    """
    exner = (p / P_0) ** (R_D / C_PD)
    theta = theta_l + (L_V / (C_PD * exner)) * q_l
    q_v = q_t - q_l
    r_total = q_t if r is None else q_t + r
    theta_v = theta * (1.0 + q_v / EPSILON) / (1.0 + r_total)
    theta_v_mean = np.mean(theta_v, axis=(1, 2), keepdims=True)
    return G * (theta_v - theta_v_mean) / theta_v_mean


@numba.njit
def _saturation_mixing_ratio(T, p):
    """
    Compute saturation mixing ratio r_s(T, p) using Clausius-Clapeyron.
    
    Uses the simplified Clausius-Clapeyron formula:
        e_s = e_s0 * exp(L_v/R_v * (1/T_0 - 1/T))
    
    Parameters:
    -----------
    T : float
        Temperature [K]
    p : float
        Pressure [Pa]
        
    Returns:
    --------
    r_s : float
        Saturation mixing ratio [kg/kg]
    """
    T_0 = 273.15      # Reference temperature [K]
    e_s0 = 611.0      # Reference saturation vapor pressure at T_0 [Pa]
    
    e_s = e_s0 * np.exp((L_V / R_V) * (1.0 / T_0 - 1.0 / T))
    r_s = EPSILON * e_s / (p - e_s)
    return r_s


@numba.njit
def _saturation_adjustment_newton(theta_l, p, r_condensable, max_iter=10, tol=1e-4):
    """
    UCLA-LES saturation adjustment using Newton-Raphson iteration.
    
    Solves for T and r_l given theta_l, p, and r_condensable (total condensable water).
    
    The target equation is:
        T = theta_l * Pi + L_v * r_l / c_pd
        
    where r_l = max(0, r_condensable - r_s(T, p))
    
    For unsaturated air (r_condensable <= r_s), r_l = 0 and T = theta_l * Pi.
    For saturated air, we iterate using Newton-Raphson with:
        f(T) = T - theta_l * Pi - L_v * r_l(T) / c_pd
        f'(T) = 1 + L_v / c_pd * dr_s/dT
        
    where dr_s/dT follows from Clausius-Clapeyron.
    
    Parameters:
    -----------
    theta_l : float
        Liquid water potential temperature [K]
    p : float
        Pressure [Pa]
    r_condensable : float
        Total condensable water mixing ratio [kg/kg] (r_t - r_rain if rain present)
    max_iter : int
        Maximum Newton-Raphson iterations
    tol : float
        Convergence tolerance for T [K]
        
    Returns:
    --------
    T : float
        Temperature [K]
    r_l : float
        Cloud liquid water mixing ratio [kg/kg]
    r_v : float
        Water vapor mixing ratio [kg/kg]
    """
    T_0 = 273.15
    e_s0 = 611.0
    
    # Exner function: Pi = (p/p_0)^(R_d/c_pd)
    kappa = R_D / C_PD
    Pi = (p / P_0) ** kappa
    
    # First guess: assume unsaturated
    T = theta_l * Pi
    r_s = _saturation_mixing_ratio(T, p)
    
    # Check if saturated
    if r_condensable <= r_s:
        # Unsaturated: no liquid water
        return T, 0.0, r_condensable
    
    # Saturated: Newton-Raphson iteration
    for _ in range(max_iter):
        r_s = _saturation_mixing_ratio(T, p)
        r_l = r_condensable - r_s
        if r_l < 0.0:
            r_l = 0.0
        
        # f(T) = T - theta_l * Pi - L_v * r_l / c_pd
        f = T - theta_l * Pi - L_V * r_l / C_PD
        
        # df/dT = 1 + L_v/c_pd * dr_s/dT
        # From Clausius-Clapeyron: dr_s/dT = r_s * L_v / (R_v * T^2) * (1 + r_s/epsilon)
        # Simplified: dr_s/dT ≈ L_v * r_s / (R_v * T^2)  for r_s << epsilon
        e_s = e_s0 * np.exp((L_V / R_V) * (1.0 / T_0 - 1.0 / T))
        dr_s_dT = (L_V * e_s * EPSILON) / (R_V * T * T * (p - e_s))
        # More accurate: include (1 + e_s/(p-e_s)) factor
        dr_s_dT = dr_s_dT * (1.0 + e_s / (p - e_s))
        
        df_dT = 1.0 + (L_V / C_PD) * dr_s_dT
        
        # Newton update
        dT = -f / df_dT
        T = T + dT
        
        if np.abs(dT) < tol:
            break
    
    # Final values
    r_s = _saturation_mixing_ratio(T, p)
    r_l = max(0.0, r_condensable - r_s)
    r_v = r_condensable - r_l
    
    return T, r_l, r_v


@numba.njit
def _density_with_loading(T, p, r_v, r_l, r_r=0.0):
    """
    Compute air density including water vapor and hydrometeor loading.
    
    Uses virtual temperature to account for vapor buoyancy and total water loading:
        T_v = T * (1 + r_v/epsilon) / (1 + r_total)
        rho = p / (R_d * T_v)
    
    This formulation accounts for:
    - Vapor reducing density (r_v/epsilon term)
    - All condensate (liquid + rain) increasing density through loading
    
    Parameters:
    -----------
    T : float
        Temperature [K]
    p : float
        Pressure [Pa]
    r_v : float
        Water vapor mixing ratio [kg/kg]
    r_l : float
        Cloud liquid water mixing ratio [kg/kg]
    r_r : float
        Rain water mixing ratio [kg/kg], default 0
        
    Returns:
    --------
    rho : float
        Air density [kg/m³]
    """
    r_total = r_v + r_l + r_r
    T_v = T * (1.0 + r_v / EPSILON) / (1.0 + r_total)
    rho = p / (R_D * T_v)
    return rho


@numba.njit
def calculate_physics_variables(p_values, theta_l_values, q_l_values, q_v_values, w_values, 
                                horizontal_resolution_squared, r_rain_values=None):
    """
    Calculate temperature, density and mass flux for cloud points using UCLA-LES thermodynamics.
    
    Uses saturation adjustment with Newton-Raphson iteration to properly diagnose
    temperature from theta_l, accounting for latent heat release in saturated air.
    
    Note: The q_l_values parameter is provided for compatibility but the actual
    vapor/liquid split is re-diagnosed from q_t = q_l + q_v using saturation adjustment.
    This ensures thermodynamic consistency.
    
    Parameters:
    -----------
    p_values : ndarray
        Pressure values [Pa]
    theta_l_values : ndarray
        Liquid water potential temperature values [K]
    q_l_values : ndarray
        Cloud liquid water content values [kg/kg] - from LES output
    q_v_values : ndarray
        Water vapor content values [kg/kg] - from LES output
    w_values : ndarray
        Vertical velocity values [m/s]
    horizontal_resolution_squared : float
        Square of horizontal resolution [m²]
    r_rain_values : ndarray or None
        Rain water mixing ratio [kg/kg], optional. If provided, rain is excluded
        from saturation adjustment but included in density loading.
        
    Returns:
    --------
    temps : ndarray
        Temperature values [K]
    rhos : ndarray
        Density values [kg/m³]
    mass_fluxes : ndarray
        Mass flux values [kg/s]
    """
    n_points = len(p_values)
    temps = np.empty(n_points)
    rhos = np.empty(n_points)
    mass_fluxes = np.empty(n_points)
    
    for i in range(n_points):
        p = p_values[i]
        theta_l = theta_l_values[i]
        q_l = q_l_values[i]
        q_v = q_v_values[i]
        w = w_values[i]
        
        # Total water (condensable for saturation adjustment)
        q_t = q_l + q_v
        
        # Rain is excluded from saturation adjustment but included in loading
        r_r = 0.0
        if r_rain_values is not None:
            r_r = r_rain_values[i]
        
        # Condensable water excludes rain (rain doesn't participate in saturation)
        r_condensable = q_t - r_r
        if r_condensable < 0.0:
            r_condensable = 0.0
        
        # Saturation adjustment to get T, r_l, r_v
        T, r_l, r_v = _saturation_adjustment_newton(theta_l, p, r_condensable)
        temps[i] = T
        
        # Density with full water loading (vapor + liquid + rain)
        rho = _density_with_loading(T, p, r_v, r_l, r_r)
        rhos[i] = rho
        
        # Mass flux
        mass_fluxes[i] = w * rho * horizontal_resolution_squared
    
    return temps, rhos, mass_fluxes

@numba.njit
def calculate_rh_and_temperature(p_values, theta_l_values, q_t_values):
    """
    Calculate Temperature and Relative Humidity assuming unsaturated air.
    
    For unsaturated environment points, q_l=0 and T = theta_l * Pi.
    
    Note: RICO data q_t is in kg/kg (not g/kg as labeled in metadata).
    
    Parameters:
    -----------
    p_values : ndarray 
        Pressure [Pa]
    theta_l_values : ndarray 
        Liquid water potential temperature [K]
    q_t_values : ndarray 
        Total water mixing ratio [kg/kg]
    
    Returns:
    --------
    temps : ndarray 
        Temperature [K]
    rhs : ndarray 
        Relative humidity [0-1]
    """
    n_points = len(p_values)
    temps = np.empty(n_points)
    rhs = np.empty(n_points)
    
    T_0 = 273.15
    e_s0 = 611.0
    kappa = R_D / C_PD
    
    for i in range(n_points):
        p = p_values[i]
        theta_l = theta_l_values[i]
        q_t = q_t_values[i]
        
        # Assume unsaturated: q_l = 0, q_v = q_t
        # For unsaturated air, T = theta_l * Pi (Exner function)
        Pi = (p / P_0) ** kappa
        T = theta_l * Pi
        temps[i] = T
        
        # Saturation vapor pressure using Clausius-Clapeyron
        e_s = e_s0 * np.exp((L_V / R_V) * (1.0 / T_0 - 1.0 / T))
        
        # Actual vapor pressure from q_v = q_t (assuming unsaturated)
        e = (q_t * p) / (EPSILON + q_t)
        
        rhs[i] = e / e_s
        
    return temps, rhs