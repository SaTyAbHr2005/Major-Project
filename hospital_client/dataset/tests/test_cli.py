
import subprocess
import os
import tempfile

def test_cli_execution(comprehensive_image_dataset):
    with tempfile.TemporaryDirectory() as out_dir:
        # call the module as a script
        result = subprocess.run([
            "python", "-m", "hospital_client.dataset", 
            "inspect", 
            os.path.join(comprehensive_image_dataset, "split_a"), 
            "--output", out_dir
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert os.path.exists(os.path.join(out_dir, "dataset_profile.json"))
        assert os.path.exists(os.path.join(out_dir, "dataset_report.md"))

