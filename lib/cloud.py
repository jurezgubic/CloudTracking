class Cloud:
    def __init__(self, cloud_id, size, surface_area, location, points, timestep):
        self.cloud_id = cloud_id
        self.size = size
        self.surface_area = surface_area
        self.location = location
        self.points = points
        self.timestep = timestep

    def __repr__(self):
        return f"Cloud(ID: {self.cloud_id}, Size: {self.size}, Surface Area: {self.surface_area}, Timestep: {self.timestep})"

