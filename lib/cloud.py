class Cloud:
    def __init__(self, cloud_id, size, surface_area, location, points, timestep, max_height, is_active=True):
        self.cloud_id = cloud_id
        self.size = size
        self.surface_area = surface_area
        self.location = location
        self.points = points
        self.timestep = timestep
        self.max_height = max_height
        self.is_active = is_active

    def update_max_height(self, new_height):
        if new_height > self.max_height:
            self.max_height = new_height

    def __repr__(self):
        return f"Cloud(ID: {self.cloud_id}, Size: {self.size}, Surface Area: {self.surface_area}, Max Height: {self.max_height}, Active: {self.is_active},  Timestep: {self.timestep})"

