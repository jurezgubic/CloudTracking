import numpy as np
import numba

@numba.njit
def calculate_physics_variables(p_values, theta_l_values, q_l_values, q_v_values, w_values, horizontal_resolution_squared):
    """Calculate temperature, density and mass flux for cloud points using JIT compilation.
    
    Parameters:
    -----------
    p_values : ndarray
        Pressure values
    theta_l_values : ndarray
        Liquid water potential temperature values
    q_l_values : ndarray
        Liquid water content values (kg/kg)
    q_v_values : ndarray
        Water vapor content values (kg/kg)
    w_values : ndarray
        Vertical velocity values (m/s)
    horizontal_resolution_squared : float
        Square of horizontal resolution (m²)
        
    Returns:
    --------
    temps : ndarray
        Temperature values (K)
    rhos : ndarray
        Density values (kg/m³)
    mass_fluxes : ndarray
        Mass flux values (kg/s)
    """
    n_points = len(p_values)
    temps = np.empty(n_points)
    rhos = np.empty(n_points)
    mass_fluxes = np.empty(n_points)
    
    # Constants (defined here for Numba compatibility)
    R_d = 287.04        # Gas constant for dry air [J/kg/K]
    R_v = 461.5         # Gas constant for water vapor [J/kg/K]
    c_pd = 1005.0       # Specific heat of dry air at constant pressure [J/kg/K]
    c_pv = 1850.0       # Specific heat of water vapor at constant pressure [J/kg/K]
    L_v = 2.5e6         # Latent heat of vaporization [J/kg]
    p_0 = 100000.0      # Reference pressure [Pa]
    epsilon = 0.622     # Ratio of gas constants for dry air and water vapor
    rho_l = 1000.0      # Density of liquid water [kg/m³]
    
    for i in range(n_points):
        # Get values for this point
        p = p_values[i]
        theta_l = theta_l_values[i]
        q_l = q_l_values[i]
        q_v = q_v_values[i]
        w = w_values[i]
        
        # Calculate kappa
        kappa = (R_d / c_pd) * ((1 + q_v / epsilon) / (1 + q_v * (c_pv / c_pd)))
        
        # Calculate temperature
        T = theta_l * (c_pd / (c_pd - L_v * q_l)) * (p_0 / p) ** (-kappa)
        temps[i] = T
        
        # Calculate density
        p_v = (q_v / (q_v + epsilon)) * p
        rho = (p - p_v) / (R_d * T) + (p_v / (R_v * T)) + (q_l * rho_l)
        rhos[i] = rho
        
        # Calculate mass flux
        mass_fluxes[i] = w * rho * horizontal_resolution_squared
    
    return temps, rhos, mass_fluxes

@numba.njit
def calculate_rh_and_temperature(p_values, theta_l_values, q_t_values):
    """
    Calculate Temperature and Relative Humidity assuming unsaturated air (q_l=0).
    
    Parameters:
    -----------
    p_values : ndarray (Pa)
    theta_l_values : ndarray (K)
    q_t_values : ndarray (kg/kg)
    
    Returns:
    --------
    temps : ndarray (K)
    rhs : ndarray (0-1)
    """
    n_points = len(p_values)
    temps = np.empty(n_points)
    rhs = np.empty(n_points)
    
    # Constants
    R_d = 287.04
    c_pd = 1005.0
    c_pv = 1850.0
    p_0 = 100000.0
    epsilon = 0.622
    
    for i in range(n_points):
        p = p_values[i]
        theta_l = theta_l_values[i]
        q_t = q_t_values[i]
        
        # Assume unsaturated: q_l = 0, q_v = q_t
        q_v = q_t
        
        # Calculate kappa
        kappa = (R_d / c_pd) * ((1 + q_v / epsilon) / (1 + q_v * (c_pv / c_pd)))
        
        # Calculate Temperature (q_l=0 simplifies the term)
        T = theta_l * (p_0 / p) ** (-kappa)
        temps[i] = T
        
        # Calculate Saturation Vapor Pressure (Magnus formula)
        # es in Pa, T in Kelvin
        T_celsius = T - 273.15
        es = 611.2 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))
        
        # Calculate Vapor Pressure
        e = (q_v * p) / (epsilon + q_v)
        
        rhs[i] = e / es
        
    return temps, rhs