import logging

import tqdm

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


import threading


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self._lock = threading.Lock()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                tqdm.tqdm.write(msg)
                self.flush()
        except Exception:
            self.handleError(record)  