import os
import logging

def setup_logging(log_filename="pipeline.log"):
    """
    Sets up the application logging to write to both stdout and a log file.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Clear existing handlers if logging was set up before
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file saved at: {log_filepath}")

def verify_paths_exist(*paths):
    """
    Verifies that the given file paths exist. Raises FileNotFoundError if any are missing.
    """
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")
