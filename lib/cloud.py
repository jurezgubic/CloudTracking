class Cloud:
    """ Create a cloud object """
    def __init__(self, cloud_id, size, surface_area, cloud_base_area, location, points, timestep, max_height, max_w, max_w_cloud_base, mass_flux, is_active=True):
        """ Initialize the cloud object """
        self.cloud_id = cloud_id
        self.size = size
        self.surface_area = surface_area
        self.cloud_base_area = cloud_base_area
        self.location = location
        self.points = points
        self.timestep = timestep
        self.max_height = max_height
        self.max_w = max_w
        self.max_w_cloud_base = max_w_cloud_base
        self.mass_flux = mass_flux
        self.is_active = is_active

    def update_max_height(self, new_height):
        """ Update the max height of the cloud """
        if new_height > self.max_height:
            self.max_height = new_height

    def __repr__(self):
        return f"Cloud(ID: {self.cloud_id}, Size: {self.size}, Surface Area: {self.surface_area}, Max Height: {self.max_height}, Max w : {max_w}, Max w cloudbase: {max_w_cloud_base}, mass flux: {mass_flux}, Active: {self.is_active},  Timestep: {self.timestep})"

