class Cloud:
    """ Create a cloud object """
    def __init__(self, 
                 cloud_id, 
                 size, 
                 surface_area, 
                 cloud_base_area, 
                 location, 
                 points, 
                 surface_points,
                 timestep, 
                 max_height, 
                 max_w, 
                 max_w_cloud_base, 
                 ql_flux, 
                 mass_flux, 
                 mass_flux_per_level, 
                 temp_per_level, 
                 theta_outside_per_level, 
                 w_per_level, 
                 circum_per_level, 
                 eff_radius_per_level, 
                 is_active=True, 
                 age=0):
        """ Initialize the cloud object """
        self.cloud_id = cloud_id
        self.size = size
        self.surface_area = surface_area
        self.cloud_base_area = cloud_base_area
        self.location = location
        self.points = points
        self.surface_points = surface_points
        self.timestep = timestep
        self.max_height = max_height
        self.max_w = max_w
        self.max_w_cloud_base = max_w_cloud_base
        self.ql_flux = ql_flux
        self.mass_flux = mass_flux
        self.mass_flux_per_level = mass_flux_per_level
        self.temp_per_level = temp_per_level
        self.theta_outside_per_level = theta_outside_per_level
        self.w_per_level = w_per_level
        self.circum_per_level = circum_per_level
        self.eff_radius_per_level = eff_radius_per_level
        self.is_active = is_active
        self.age = age
        self.merged_into = None  # Track ID this cloud merged into, if any

    def update_max_height(self, new_height):
        """ Update the max height of the cloud """
        if new_height > self.max_height:
            self.max_height = new_height

    def __repr__(self):
        return f"Cloud(ID: {self.cloud_id}, Size: {self.size}, Surface Area: {self.surface_area}, Max Height: {self.max_height}, Max w : {max_w}, Max w cloudbase: {max_w_cloud_base}, mass flux: {mass_flux}, Active: {self.is_active},  Timestep: {self.timestep})"

