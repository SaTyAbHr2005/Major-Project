"""
Hospital-local HTTP bridge between the Hospital Desktop UI and the Member 1
ML pipeline (Modules 4, 5, 6, 7, 8, 16). It runs the SAME module CLIs a user
runs in the terminal, as subprocesses, and serves their output files.

Binds to 127.0.0.1 only: raw images and datasets never leave this machine.
Run with:  python -m hospital_client.local_api
"""
