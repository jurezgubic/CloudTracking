"""Tests for MONC configuration file parser."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.config_parsers.mcf_parser import MONCConfigParser


class TestMONCConfigParser:
    """Tests for the MONCConfigParser class."""
    
    @pytest.fixture
    def sample_mcf_content(self):
        """Sample MCF file content for testing."""
        return """
# Global configuration
global_configuration=global_config

# Booleans
l_thoff=.true.
use_anelastic_equations=.true.
passive_th=.false.

# Grid configuration
x_size=256
y_size=256
z_size=211
dxx=200
dyy=200
zztop=23000.0

# Pressure configuration
surface_pressure=99130.
surface_reference_pressure=100000.

# Reference theta profile
l_init_pl_theta=.true.
z_init_pl_theta=0.0,334.0,443.0,970.0,1523.0
f_init_pl_theta=297.59,300.45,300.81,303.27,305.76

# String values
ioserver_configuration_file="io/io_cfg_files/data_write.xml"
varying_theta_coordinate="height"

# Comment after value
termination_time=21600. # 6 hours
"""
    
    @pytest.fixture
    def mcf_file(self, sample_mcf_content, tmp_path):
        """Create a temporary MCF file for testing."""
        mcf_path = tmp_path / "test_config.mcf"
        mcf_path.write_text(sample_mcf_content)
        return mcf_path
    
    def test_parse_basic(self, mcf_file):
        """Test basic parsing of MCF file."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        assert 'x_size' in config
        assert 'dxx' in config
        assert 'surface_pressure' in config
    
    def test_parse_integers(self, mcf_file):
        """Test parsing of integer values."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        assert config['x_size'] == 256
        assert config['y_size'] == 256
        assert config['z_size'] == 211
        assert isinstance(config['x_size'], int)
    
    def test_parse_floats(self, mcf_file):
        """Test parsing of float values."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        assert config['dxx'] == 200
        assert config['dyy'] == 200
        assert config['surface_pressure'] == 99130.0
        assert config['surface_reference_pressure'] == 100000.0
        assert config['zztop'] == 23000.0
    
    def test_parse_booleans(self, mcf_file):
        """Test parsing of Fortran boolean values."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        assert config['l_thoff'] is True
        assert config['use_anelastic_equations'] is True
        assert config['passive_th'] is False
    
    def test_parse_arrays(self, mcf_file):
        """Test parsing of comma-separated arrays."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        z_theta = config['z_init_pl_theta']
        f_theta = config['f_init_pl_theta']
        
        assert isinstance(z_theta, np.ndarray)
        assert isinstance(f_theta, np.ndarray)
        assert len(z_theta) == 5
        assert len(f_theta) == 5
        
        np.testing.assert_array_almost_equal(
            z_theta, 
            [0.0, 334.0, 443.0, 970.0, 1523.0]
        )
        np.testing.assert_array_almost_equal(
            f_theta,
            [297.59, 300.45, 300.81, 303.27, 305.76]
        )
    
    def test_parse_strings(self, mcf_file):
        """Test parsing of quoted string values."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        assert config['ioserver_configuration_file'] == "io/io_cfg_files/data_write.xml"
        assert config['varying_theta_coordinate'] == "height"
    
    def test_parse_comments_stripped(self, mcf_file):
        """Test that inline comments are properly stripped."""
        parser = MONCConfigParser(mcf_file)
        config = parser.parse()
        
        # termination_time should be 21600, not "21600. # 6 hours"
        assert config['termination_time'] == 21600.0
    
    def test_get_reference_theta_profile(self, mcf_file):
        """Test get_reference_theta_profile helper method."""
        parser = MONCConfigParser(mcf_file)
        parser.parse()
        
        z, f = parser.get_reference_theta_profile()
        
        assert len(z) == 5
        assert len(f) == 5
        assert z[0] == 0.0
        assert f[0] == 297.59
    
    def test_get_grid_spacing(self, mcf_file):
        """Test get_grid_spacing helper method."""
        parser = MONCConfigParser(mcf_file)
        parser.parse()
        
        dx, dy = parser.get_grid_spacing()
        
        assert dx == 200.0
        assert dy == 200.0
    
    def test_get_surface_pressures(self, mcf_file):
        """Test get_surface_pressures helper method."""
        parser = MONCConfigParser(mcf_file)
        parser.parse()
        
        p_surf, p_ref = parser.get_surface_pressures()
        
        assert p_surf == 99130.0
        assert p_ref == 100000.0
    
    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        fake_path = tmp_path / "nonexistent.mcf"
        
        with pytest.raises(FileNotFoundError):
            MONCConfigParser(fake_path)
    
    def test_missing_required_params(self, tmp_path):
        """Test that ValueError is raised when required params are missing."""
        incomplete_content = """
x_size=256
y_size=256
"""
        mcf_path = tmp_path / "incomplete.mcf"
        mcf_path.write_text(incomplete_content)
        
        parser = MONCConfigParser(mcf_path)
        
        with pytest.raises(ValueError) as exc_info:
            parser.parse()
        
        assert "Missing required parameters" in str(exc_info.value)
    
    def test_get_method(self, mcf_file):
        """Test the get() method with default values."""
        parser = MONCConfigParser(mcf_file)
        parser.parse()
        
        # Existing key
        assert parser.get('x_size') == 256
        
        # Non-existing key with default
        assert parser.get('nonexistent', 'default') == 'default'
        
        # Non-existing key without default
        assert parser.get('nonexistent') is None


class TestMONCConfigParserWithRealFile:
    """Tests using the actual LBA config file if available."""
    
    @pytest.fixture
    def real_mcf_file(self):
        """Path to the real LBA config file."""
        path = Path(__file__).parent.parent / "lba_config.mcf"
        if not path.exists():
            pytest.skip("lba_config.mcf not found in project root")
        return path
    
    def test_parse_real_file(self, real_mcf_file):
        """Test parsing the actual LBA configuration file."""
        parser = MONCConfigParser(real_mcf_file)
        config = parser.parse()
        
        # Check key parameters exist
        assert 'dxx' in config
        assert 'dyy' in config
        assert 'surface_pressure' in config
        assert 'surface_reference_pressure' in config
        assert 'z_init_pl_theta' in config
        assert 'f_init_pl_theta' in config
        
        # Check values make physical sense
        assert config['dxx'] > 0
        assert config['surface_pressure'] > 90000  # ~900 hPa
        assert config['surface_pressure'] < 110000  # ~1100 hPa
        
        # Check theta profile
        z, f = parser.get_reference_theta_profile()
        assert len(z) > 10  # Should have many levels
        assert z[0] == 0.0  # Should start at surface
        assert f[0] > 290 and f[0] < 320  # Reasonable surface theta
