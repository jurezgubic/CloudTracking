class Cloud:
    """Create a cloud object."""
    def __init__(self,
                 cloud_id,
                 size,
                 surface_area,
                 cloud_base_area,
                 cloud_base_height,
                 location,
                 points,
                 surface_points,
                 timestep,
                 max_height,
                 max_w,
                 max_w_cloud_base,
                 mean_u,
                 mean_v,
                 mean_w,
                 ql_flux,
                 mass_flux,
                 mass_flux_per_level,
                 temp_per_level,
                 theta_outside_per_level,
                 w_per_level,
                 circum_per_level,
                 eff_radius_per_level,  # legacy: stored compactness ratio
                 # Per-level kinematics (for NIP)
                 u_per_level=None,
                 v_per_level=None,
                 # NIP diagnostics
                 nip_per_level=None,
                 nip_acc_per_level=None,
                 is_active=True,
                 age=0,
                 area_per_level=None,
                 equiv_radius_per_level=None,
                 compactness_per_level=None,
                 base_radius_diagnosed=None,
                 base_area_diagnosed=None,
                 max_equiv_radius=None,
                 # Environment Aloft
                 env_aloft_qt_diff=None,
                 env_aloft_thetal_diff=None,
                 env_aloft_shear=None,
                 env_aloft_n2=None,
                 env_aloft_rh=None,
                 env_aloft_qt_std=None,
                 env_aloft_thetal_std=None,
                 env_aloft_shear_std=None,
                 env_aloft_n2_std=None,
                 env_aloft_rh_std=None):
        """ Initialize the cloud object """
        self.cloud_id = cloud_id
        self.size = size
        self.surface_area = surface_area
        self.cloud_base_area = cloud_base_area
        self.cloud_base_height = cloud_base_height  # Add new property
        self.location = location
        self.points = points
        self.surface_points = surface_points
        self.timestep = timestep
        self.max_height = max_height
        self.max_w = max_w
        self.max_w_cloud_base = max_w_cloud_base
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.mean_w = mean_w
        self.ql_flux = ql_flux
        self.mass_flux = mass_flux
        self.mass_flux_per_level = mass_flux_per_level
        self.temp_per_level = temp_per_level
        self.theta_outside_per_level = theta_outside_per_level
        self.w_per_level = w_per_level
        self.circum_per_level = circum_per_level
        self.eff_radius_per_level = eff_radius_per_level  # legacy (compactness ratio)
        # Per-level kinematics needed by NIP
        self.u_per_level = u_per_level
        self.v_per_level = v_per_level
        # NIP instantaneous and accumulated
        self.nip_per_level = nip_per_level
        self.nip_acc_per_level = nip_acc_per_level
        self.is_active = is_active
        self.age = age
        self.merged_into = None  # Track ID this cloud merged into, if any

        # Merges and splits
        self.merges_count = 0      # Number of times this cloud has merged with others
        self.splits_count = 0      # Number of times this cloud has split from others
        self.merged_with = []      # List of cloud IDs this cloud has merged with
        self.split_from = None     # Cloud ID this cloud split from (if any)

        self.area_per_level = area_per_level
        self.equiv_radius_per_level = equiv_radius_per_level
        self.compactness_per_level = compactness_per_level
        self.base_radius_diagnosed = base_radius_diagnosed
        self.base_area_diagnosed = base_area_diagnosed
        self.max_equiv_radius = max_equiv_radius

        # Environment Aloft
        self.env_aloft_qt_diff = env_aloft_qt_diff
        self.env_aloft_thetal_diff = env_aloft_thetal_diff
        self.env_aloft_shear = env_aloft_shear
        self.env_aloft_n2 = env_aloft_n2
        self.env_aloft_rh = env_aloft_rh
        self.env_aloft_qt_std = env_aloft_qt_std
        self.env_aloft_thetal_std = env_aloft_thetal_std
        self.env_aloft_shear_std = env_aloft_shear_std
        self.env_aloft_n2_std = env_aloft_n2_std
        self.env_aloft_rh_std = env_aloft_rh_std

        # Environment rings (per level x ring distance)
        # Averages over environment points at Manhattan distance d=1..D from cloud edge
        self.env_w_rings = None
        self.env_l_rings = None
        self.env_qt_rings = None
        self.env_qv_rings = None
        self.env_p_rings = None
        self.env_theta_l_rings = None
        self.env_buoyancy_rings = None

    def update_max_height(self, new_height):
        """ Update the max height of the cloud """
        if new_height > self.max_height:
            self.max_height = new_height

    def __repr__(self):
        return f"Cloud(ID: {self.cloud_id}, Size: {self.size}, Surface Area: {self.surface_area}, Max Height: {self.max_height}, Max w: {self.max_w}, Max w cloudbase: {self.max_w_cloud_base}, mass flux: {self.mass_flux}, Active: {self.is_active}, Timestep: {self.timestep})"
