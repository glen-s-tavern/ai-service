import logging
import tqdm
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