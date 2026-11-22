from scripts.run_grid import validate_config


def test_validate_config_returns_enabled_algorithms():
    config = {
        "algorithms": {
            "fedavg": {"enabled": True},
            "fedprox": {"enabled": True},
            "scaffold": {"enabled": False},
        },
        "data_distributions": [
            {"name": "iid", "partition": "iid"}
        ],
        "seeds": [0, 1],
        "training": {"num_rounds": 100},
        "output": {"results_dir": "results"},
    }

    algorithms = validate_config(config)

    assert algorithms == ["fedavg", "fedprox"]


def test_validate_config_rejects_no_enabled_algorithms():
    config = {
        "algorithms": {
            "fedavg": {"enabled": False},
            "fedprox": {"enabled": False},
        },
        "data_distributions": [
            {"name": "iid", "partition": "iid"}
        ],
        "seeds": [0],
        "training": {"num_rounds": 100},
        "output": {"results_dir": "results"},
    }

    import pytest

    with pytest.raises(ValueError, match="No algorithms are enabled"):
        validate_config(config)