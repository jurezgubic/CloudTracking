import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# Constants
rho_air = 1.25  # kg/m^3 density of air at sea level
L_v = 2.5*10e6 # J/kg from UCLA-LES documentation
c_p = 1004.0  # J/kg/K
R_d = 287.05  # J/kg/K gas constant for dry air
R_v = 461.51  # J/kg/K gas constant for water vapor
p_0 = 100000.0  # Pa standard pressure at sea level
c_pd = 1004.0  # J/kg/K specific heat capacity of dry air
c_pv = 1996.0  # J/kg/K specific heat capacity of water vapor
rho_water = 1000.0  # kg/m^3 density of water
epsilon = R_d / R_v

# File paths
w_file = '/Users/jure/PhD/coding/RICO_1hr/rico.w.nc'
l_file = '/Users/jure/PhD/coding/RICO_1hr/rico.l.nc'
q_file = '/Users/jure/PhD/coding/RICO_1hr/rico.q.nc'
t_file = '/Users/jure/PhD/coding/RICO_1hr/rico.t.nc'
p_file = '/Users/jure/PhD/coding/RICO_1hr/rico.p.nc'

def load_data(timestep):
    with nc.Dataset(w_file) as w_ds, \
         nc.Dataset(l_file) as l_ds, \
         nc.Dataset(q_file) as q_ds, \
         nc.Dataset(t_file) as t_ds, \
         nc.Dataset(p_file) as p_ds:

        w_data = w_ds['w'][timestep, :, :, :]
        l_data = l_ds['l'][timestep, :, :, :] / 1000
        q_data = q_ds['q'][timestep, :, :, :] / 1000
        t_data = t_ds['t'][timestep, :, :, :]
        p_data = p_ds['p'][timestep, :, :, :]

    return w_data, l_data, q_data, t_data, p_data

def load_zt_values():
    with nc.Dataset(l_file) as ds:
        zt_values = ds['zt'][:]
    return zt_values

def calculate_theta_through_theta_l(t_data, l_data):
    theta = t_data * (c_pd / (c_pd - l_data * L_v))
    return theta

def calculate_theta_through_T(t_data, p_data, q_data, l_data):
    q_v_data = (q_data - l_data)
    kappa = (R_d / c_pd) * ((1 + q_v_data / epsilon) / (1 + q_v_data * (c_pv / c_pd)))
    T = t_data * (c_pd / (c_pd - L_v * l_data)) * (p_0 / p_data) ** (-kappa)
    theta = T * (p_0 / p_data) ** kappa
    return theta

def calculate_m_c_values(w_data, l_data, variable_data, liquid_water_threshold, multiplier):
    m_c_values = []
    active_cloudy_values = []
    environment_values = []
    total_flux_values = []

    for z in range(w_data.shape[0]):
        # Cloudy and environment masks
        cloudy_mask = l_data[z, :, :] > liquid_water_threshold
        environment_mask = ~cloudy_mask
        area_fraction = np.mean(cloudy_mask) # area fraction

        # Mean vertical velocities and variable contents
        w_c_mean = np.mean(w_data[z, :, :][cloudy_mask])
        w_e_mean = np.mean(w_data[z, :, :][environment_mask])
        var_c_mean = np.mean(variable_data[z, :, :][cloudy_mask])
        var_e_mean = np.mean(variable_data[z, :, :][environment_mask])

        # Perturbations (variable minus its mean)
        w_prime_c = np.nan_to_num(w_data[z, :, :][cloudy_mask] - w_c_mean)
        w_prime_e = np.nan_to_num(w_data[z, :, :][environment_mask] - w_e_mean)
        var_prime_c = np.nan_to_num(variable_data[z, :, :][cloudy_mask] - var_c_mean)
        var_prime_e = np.nan_to_num(variable_data[z, :, :][environment_mask] - var_e_mean)

        # Calculate: a * <w''variable''>^c
        term1 = area_fraction * np.mean(w_prime_c * var_prime_c) * multiplier
        # Calculate: (1 - a) * <w''variable''>^e
        term2 = (1 - area_fraction) * np.mean(w_prime_e * var_prime_e) * multiplier

        # Calculate m_c
        m_c = area_fraction * (1 - area_fraction) * (w_c_mean - w_e_mean)
        # Multiply m_c with (var_c_mean - var_e_mean)
        term3 = m_c * (var_c_mean - var_e_mean) * multiplier

        # Append the result for the current vertical level
        m_c_values.append(term3)
        active_cloudy_values.append(term1)
        environment_values.append(term2)

        # Calculate total flux by summing valid numbers only
        total_flux = 0
        if np.isfinite(term3):
            total_flux += term3
        if np.isfinite(term1):
            total_flux += term1
        if np.isfinite(term2):
            total_flux += term2

        total_flux_values.append(total_flux)

    return np.array(m_c_values), np.array(active_cloudy_values), np.array(environment_values), np.array(total_flux_values)

def calculate_m_c_for_timestep(timestep, variable, multiplier, liquid_water_threshold):
    w_data, l_data, q_data, t_data, p_data = load_data(timestep)

    if variable == 'theta_l':
        variable_data = t_data
    elif variable == 'q_t':
        variable_data = q_data
    elif variable == 'q_l':
        variable_data = l_data
    elif variable == 'theta':
        variable_data = calculate_theta_through_theta_l(t_data, l_data)
    else:
        raise ValueError("Unknown variable")
    return calculate_m_c_values(w_data, l_data, variable_data, liquid_water_threshold, multiplier)



def calculate_percentage_contribution(m_c_values, total_flux_values):
    percentage_contribution = (m_c_values / total_flux_values) * 100
    return percentage_contribution



def plot_with_additional_lines(ax, m_c_values, active_cloudy_values, environment_values, total_flux_values, title, zt_values):
    ax.plot(m_c_values, zt_values, label='$M_c(\\overline{\phi}^c - \\overline{\phi}^e)$')
    ax.plot(active_cloudy_values, zt_values, label='a($\\overline{w\'\'\phi\'\'}^c$)')
    ax.plot(environment_values, zt_values, label='(1-a)($\\overline{w\'\'\phi\'\'}^e$)')
    ax.plot(total_flux_values, zt_values, label='Total Flux')

    ax.set_xlabel('W/m^2')
    ax.set_ylabel('Height (m)')
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim([0, 2000])
    ax.legend()


