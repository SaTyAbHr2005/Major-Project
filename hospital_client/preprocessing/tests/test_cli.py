import sys
import pytest
from unittest.mock import patch
from hospital_client.preprocessing.cli import main

@patch("hospital_client.preprocessing.cli.PreprocessingEngine")
@patch("sys.argv", ["__main__.py", "preprocess", "dummy_data", "--profile", "dummy_profile.json", "--output", "out_dir", "--mode", "lazy"])
def test_cli_success(mock_engine):
    main()
    mock_engine.assert_called_once()
    mock_engine.return_value.run.assert_called_with("dummy_profile.json")
    
@patch("hospital_client.preprocessing.cli.PreprocessingEngine")
@patch("sys.argv", ["__main__.py", "preprocess", "dummy_data", "--profile", "dummy_profile.json", "--output", "out_dir", "--mode", "materialized"])
def test_cli_materialized(mock_engine):
    main()
    mock_engine.assert_called_once()
    
@patch("hospital_client.preprocessing.cli.PreprocessingEngine")
@patch("sys.argv", ["__main__.py", "preprocess", "dummy_data", "--profile", "dummy_profile.json", "--output", "out_dir"])
def test_cli_exception(mock_engine):
    mock_engine.return_value.run.side_effect = ValueError("Some error")
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1

@patch("hospital_client.preprocessing.cli.argparse.ArgumentParser.print_help")
@patch("sys.argv", ["__main__.py"])
def test_cli_no_args(mock_print_help):
    main()
    mock_print_help.assert_called_once()

