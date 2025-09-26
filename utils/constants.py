import numpy as np

# Constants
R_d = 287.05  # J/kg/K gas constant for dry air
R_v = 461.51  # J/kg/K gas constant for water vapor
c_pd = 1004.0  # J/kg/K specific heat capacity of dry air
c_pv = 1996.0  # J/kg/K specific heat capacity of water vapour
L_v = 2268000.0  # J/kg latent heat of vaporisation
p_0 = 100000.0  # Pa standard pressure at sea level
rho_l = 1000.0  # kg/m^3 density of water


# Composite constants
epsilon = R_d / R_v

