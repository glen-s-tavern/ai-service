import logging
from tqdm_logger_handler import TqdmLoggingHandler

def setup_logger(name):
    # Create a logger specific to the module
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Only set basicConfig if it hasn't been configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
    
    # Add TqdmLoggingHandler
    logger.addHandler(TqdmLoggingHandler())
    
    return logger 