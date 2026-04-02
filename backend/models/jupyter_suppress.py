import warnings
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", message=".*PyDataset.*super.*")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
