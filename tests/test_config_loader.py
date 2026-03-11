import pytest

from src.config_loader import _flatten, load_config


@pytest.fixture
def ucla_toml(tmp_path):
    """Create a minimal UCLA-LES TOML config pointing at tmp_path as data dir."""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(f"""\
[data]
data_format = "UCLA-LES"
base_file_path = "{tmp_path}"

[data.file_name]
l = "rico.l.nc"
w = "rico.w.nc"

[cloud_identification]
min_size = 10
l_condition = 5e-4

[simulation]
timestep_duration = 60
horizontal_resolution = 25.0
""")
    return str(toml_file)


@pytest.fixture
def monc_toml(tmp_path):
    """Create a minimal MONC TOML config with real temp paths."""
    data_dir = tmp_path / "monc_data"
    data_dir.mkdir()
    config_file = data_dir / "config.mcf"
    config_file.write_text("# dummy mcf")
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(f"""\
[data]
data_format = "MONC"
monc_data_path = "{data_dir}"
monc_config_file = "{config_file}"
monc_file_pattern = "3dfields_ts_{{time}}.nc"

[cloud_identification]
min_size = 10
""")
    return str(toml_file)


class TestLoadConfig:
    def test_load_ucla_config(self, ucla_toml):
        config = load_config(ucla_toml)
        assert config["data_format"] == "UCLA-LES"
        assert config["min_size"] == 10
        assert config["l_condition"] == 5e-4
        assert config["timestep_duration"] == 60

    def test_load_monc_config(self, monc_toml):
        config = load_config(monc_toml)
        assert config["data_format"] == "MONC"
        assert "monc_data_path" in config
        assert "monc_config_file" in config

    def test_file_name_stays_nested(self, ucla_toml):
        config = load_config(ucla_toml)
        assert isinstance(config["file_name"], dict)
        assert config["file_name"]["l"] == "rico.l.nc"

    def test_sections_are_flattened(self, ucla_toml):
        config = load_config(ucla_toml)
        # Keys from [cloud_identification] and [simulation] should be top-level
        assert "min_size" in config
        assert "horizontal_resolution" in config
        # Section names should not be keys
        assert "cloud_identification" not in config
        assert "simulation" not in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.toml")

    def test_missing_data_format_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[cloud_identification]\nmin_size = 10\n')
        with pytest.raises(ValueError, match="Missing required config keys"):
            load_config(str(toml_file))

    def test_unknown_data_format_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[data]\ndata_format = "UNKNOWN"\n')
        with pytest.raises(ValueError, match="Unknown data_format"):
            load_config(str(toml_file))

    def test_missing_ucla_keys_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[data]\ndata_format = "UCLA-LES"\n')
        with pytest.raises(ValueError, match="UCLA-LES format requires"):
            load_config(str(toml_file))

    def test_missing_monc_keys_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[data]\ndata_format = "MONC"\n')
        with pytest.raises(ValueError, match="MONC format requires"):
            load_config(str(toml_file))

    def test_nonexistent_data_path_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("""\
[data]
data_format = "UCLA-LES"
base_file_path = "/nonexistent/path"

[data.file_name]
l = "rico.l.nc"
""")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_config(str(toml_file))


class TestFlatten:
    def test_flat_input_unchanged(self):
        raw = {"a": 1, "b": "hello"}
        assert _flatten(raw) == {"a": 1, "b": "hello"}

    def test_nested_sections_flattened(self):
        raw = {"section": {"key1": 1, "key2": 2}}
        assert _flatten(raw) == {"key1": 1, "key2": 2}

    def test_file_name_preserved(self):
        raw = {"data": {"file_name": {"l": "f.nc"}}}
        result = _flatten(raw)
        assert result["file_name"] == {"l": "f.nc"}
