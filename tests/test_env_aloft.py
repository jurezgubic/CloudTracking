import unittest
import numpy as np
from lib.cloudfield import CloudField
import utils.constants as const

class TestEnvAloft(unittest.TestCase):
    """Test the environment aloft analysis in CloudField."""

    def setUp(self):
        # Grid dimensions
        self.nz, self.ny, self.nx = 20, 20, 20
        self.xt = np.arange(self.nx) * 25.0
        self.yt = np.arange(self.ny) * 25.0
        self.zt = np.arange(self.nz) * 25.0
        
        # Configuration
        self.config = {
            'distance_threshold': 100.0,
            'w_switch': False,
            'l_condition': 0.5,
            'min_size': 1,
            'horizontal_resolution': 25.0,
            'plot_switch': False,
            'cloud_base_altitude': 0.0,
            'env_aloft_levels': 5,  # Test with 5 levels aloft
            'env_ring_max_distance': 1,
            'env_periodic_rings': False,
            'match_shell_layers': 1
        }
        
        # Initialize data arrays
        self.l_data = np.zeros((self.nz, self.ny, self.nx), dtype=int)
        self.u_data = np.zeros((self.nz, self.ny, self.nx))
        self.v_data = np.zeros((self.nz, self.ny, self.nx))
        self.w_data = np.zeros((self.nz, self.ny, self.nx))
        self.p_data = np.full((self.nz, self.ny, self.nx), 100000.0)
        self.theta_l_data = np.full((self.nz, self.ny, self.nx), 300.0)
        self.q_t_data = np.zeros((self.nz, self.ny, self.nx))
        
        # Note: RICO LES data water species are in kg/kg (not g/kg as metadata labels)
        # Tests use small values appropriate for kg/kg units

    def test_env_aloft_extraction(self):
        """Test that environment variables are correctly extracted above the cloud."""
        
        # 1. Create a simple block cloud
        # Cloud from z=2 to z=5, covering y=5-10, x=5-10
        z_top_idx = 5
        self.l_data[2:z_top_idx+1, 5:10, 5:10] = 1
        
        # 2. Set up environment properties ABOVE the cloud
        # We are testing 5 levels aloft.
        # Level 1 aloft is at z = z_top_idx + 1 = 6
        # Level 2 aloft is at z = 7, etc.
        
        # --- Test QT Diff ---
        # Set background q_t to 0.001 kg/kg (1 g/kg)
        self.q_t_data[:] = 0.001 
        # Set q_t at aloft level 1 (z=6) to 0.002 kg/kg (2 g/kg)
        # Expected diff: 0.002 - domain_mean (slightly affected by anomaly)
        self.q_t_data[z_top_idx+1, 5:10, 5:10] = 0.002
        
        # --- Test Shear ---
        # Shear is calculated using u, v at z+1 and z-1 (or similar) relative to the aloft level.
        # Let's test shear at aloft level 2 (z=7).
        # Aloft level 2 is at z=7.
        # It uses neighbors. If z=7, it likely uses z=8 and z=6.
        # Let's set a simple shear in U.
        # u at z=6 is 0.
        # u at z=8 is 10.0.
        # dz between 6 and 8 is 2 * 25 = 50m.
        # du/dz = 10 / 50 = 0.2 s^-1.
        self.u_data[z_top_idx+1, 5:10, 5:10] = 0.0   # z=6
        self.u_data[z_top_idx+3, 5:10, 5:10] = 10.0  # z=8
        
        # --- Test Theta_l Diff ---
        # Background theta_l = 300
        # Set theta_l at aloft level 3 (z=8) to 305.
        # Expected diff: 5.0 K
        self.theta_l_data[z_top_idx+3, 5:10, 5:10] = 305.0
        
        # 3. Run CloudField
        timestep = 0
        cf = CloudField(
            self.l_data, self.u_data, self.v_data, self.w_data, 
            self.p_data, self.theta_l_data, self.q_t_data, 
            timestep, self.config, self.xt, self.yt, self.zt
        )
        
        # 4. Verify Results
        self.assertEqual(len(cf.clouds), 1)
        cloud_id = list(cf.clouds.keys())[0]
        cloud = cf.clouds[cloud_id]
        
        # Check dimensions
        self.assertEqual(len(cloud.env_aloft_qt_diff), 5)
        
        # Check QT Diff at Level 0 (z=6)
        # Note: The domain mean is affected by our anomaly.
        # Domain size 20x20 = 400 pixels.
        # Anomaly area 5x5 = 25 pixels.
        # Background = 0.001 kg/kg, Anomaly = 0.002 kg/kg.
        # Mean = (375*0.001 + 25*0.002)/400 = 0.0010625 kg/kg
        # Diff = 0.002 - 0.0010625 = 0.0009375 kg/kg
        expected_qt_diff = 0.0009375
        self.assertAlmostEqual(cloud.env_aloft_qt_diff[0], expected_qt_diff, places=6)
        
        # Check Shear at Level 1 (z=7)
        # We set u at z=6 to 0 and z=8 to 10.
        # At z=7, it uses z=8 and z=6.
        # du/dz = (10 - 0) / (2*25) = 0.2
        expected_shear = 0.2
        self.assertAlmostEqual(cloud.env_aloft_shear[1], expected_shear, places=6)
        
        # Check Theta_l Diff at Level 2 (z=8)
        # We set theta_l at z=8 to 305. Mean is 300.
        # Mean = (375*300 + 25*305)/400 = 300.3125
        # Diff = 305 - 300.3125 = 4.6875
        expected_thetal_diff = 4.6875
        self.assertAlmostEqual(cloud.env_aloft_thetal_diff[2], expected_thetal_diff, places=6)

    def test_variable_cloud_top(self):
        """Test that extraction follows the cloud top topography."""
        # Create a cloud with a step in it
        # Left half (x=5-7): top at z=5
        # Right half (x=8-10): top at z=7
        self.l_data[2:6, 5:10, 5:8] = 1  # Top at 5
        self.l_data[2:8, 5:10, 8:11] = 1 # Top at 7
        
        # We want to test that "Level 1 aloft" means z=6 for left half and z=8 for right half.
        
        # Set a marker value in q_t (in kg/kg units)
        # At z=6 (aloft 1 for left), set q_t = 0.002 kg/kg (2 g/kg)
        self.q_t_data[6, 5:10, 5:8] = 0.002
        
        # At z=8 (aloft 1 for right), set q_t = 0.003 kg/kg (3 g/kg)
        self.q_t_data[8, 5:10, 8:11] = 0.003
        
        # Background is 0.0
        
        # Run CloudField
        timestep = 0
        cf = CloudField(
            self.l_data, self.u_data, self.v_data, self.w_data, 
            self.p_data, self.theta_l_data, self.q_t_data, 
            timestep, self.config, self.xt, self.yt, self.zt
        )
        
        cloud = list(cf.clouds.values())[0]
        
        # Test that the qt_diff is positive (indicating we sampled the anomaly region)
        # and within the expected order of magnitude for our kg/kg anomalies
        self.assertGreater(cloud.env_aloft_qt_diff[0], 0.001)  # Should be > 0.001 kg/kg
        self.assertLess(cloud.env_aloft_qt_diff[0], 0.003)     # Should be < 0.003 kg/kg

if __name__ == '__main__':
    unittest.main()