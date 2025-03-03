
# filepath: /Users/jure/PhD/coding/tracking/cloudtracker/tests/test_main.py
import unittest
from unittest.mock import patch, MagicMock
from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker
from main import finalize_partial_lifetime_tracks

class TestMainPartialLifetime(unittest.TestCase):
    @patch("main.Dataset")
    def test_finalize_partial_lifetime_tracks(self, mock_dataset_cls):
        """
        Test that partial-lifetime clouds are flagged as invalid in the NetCDF file.
        """
        # Mock a netCDF Dataset instance
        mock_dataset = MagicMock()
        mock_dataset.variables = {"valid_track": [1, 1, 1]}
        mock_dataset_cls.return_value.__enter__.return_value = mock_dataset

        # Create a mock CloudTracker with 3 tracks: 
        # 1. Partial lifetime (starts at t=0)
        # 2. Partial lifetime (still active at t=19 if total_timesteps=20)
        # 3. Full lifetime (starts at t=1, ends at t=18, is_inactive by then)
        cloud_tracker = CloudTracker(config={})
        cloud_tracker.cloud_tracks = {
            0: [Cloud(cloud_id=0, size=10, surface_area=5, cloud_base_area=5, location=(0,0,0),
                      points=[], timestep=0, max_height=500, max_w=1, max_w_cloud_base=0.5, 
                      ql_flux=0, mass_flux=0, mass_flux_per_level=[], temp_per_level=[], 
                      theta_outside_per_level=[], w_per_level=[], circum_per_level=[], 
                      eff_radius_per_level=[], is_active=False)],
            1: [Cloud(cloud_id=1, size=10, surface_area=5, cloud_base_area=5, location=(0,0,0),
                      points=[], timestep=5, max_height=500, max_w=1, max_w_cloud_base=0.5, 
                      ql_flux=0, mass_flux=0, mass_flux_per_level=[], temp_per_level=[], 
                      theta_outside_per_level=[], w_per_level=[], circum_per_level=[], 
                      eff_radius_per_level=[], is_active=True),
                Cloud(cloud_id=1, size=10, surface_area=5, cloud_base_area=5, location=(0,0,0),
                      points=[], timestep=19, max_height=500, max_w=1, max_w_cloud_base=0.5, 
                      ql_flux=0, mass_flux=0, mass_flux_per_level=[], temp_per_level=[], 
                      theta_outside_per_level=[], w_per_level=[], circum_per_level=[], 
                      eff_radius_per_level=[], is_active=True)],
            2: [Cloud(cloud_id=2, size=10, surface_area=5, cloud_base_area=5, location=(0,0,0),
                      points=[], timestep=1, max_height=500, max_w=1, max_w_cloud_base=0.5, 
                      ql_flux=0, mass_flux=0, mass_flux_per_level=[], temp_per_level=[], 
                      theta_outside_per_level=[], w_per_level=[], circum_per_level=[], 
                      eff_radius_per_level=[], is_active=True),
                Cloud(cloud_id=2, size=10, surface_area=5, cloud_base_area=5, location=(0,0,0),
                      points=[], timestep=18, max_height=500, max_w=1, max_w_cloud_base=0.5, 
                      ql_flux=0, mass_flux=0, mass_flux_per_level=[], temp_per_level=[], 
                      theta_outside_per_level=[], w_per_level=[], circum_per_level=[], 
                      eff_radius_per_level=[], is_active=False)]
        }

        # total_timesteps = 20 => track #0 partial-lifetime (started at t=0),
        # track #1 partial-lifetime (still active at t=19),
        # track #2 is valid (started after t=0 and ended before t=19).
        finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps=20)

        # Check the calls to set valid_track = 0
        self.assertIn(0, cloud_tracker.cloud_tracks)
        self.assertIn(1, cloud_tracker.cloud_tracks)
        self.assertIn(2, cloud_tracker.cloud_tracks)

        # valid_track array in mock netcdf was initially [1,1,1]
        # We expect track 0 and track 1 to be set to 0
        # track 2 remains 1
        # Because we pass them in enumerate order: t_id=[0,1,2].
        self.assertEqual(mock_dataset.variables["valid_track"][0], 0, "Track 0 should be flagged as partial-lifetime")
        self.assertEqual(mock_dataset.variables["valid_track"][1], 0, "Track 1 should be flagged as partial-lifetime")
        self.assertEqual(mock_dataset.variables["valid_track"][2], 1, "Track 2 should remain valid")

if __name__ == '__main__':
    unittest.main()