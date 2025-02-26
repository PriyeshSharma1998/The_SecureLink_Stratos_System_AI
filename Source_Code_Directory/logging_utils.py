import logging

class LoggerUtility:
    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Set up the logger and specify log level and handlers."""
        try:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler_stream = logging.StreamHandler()
            handler_file = logging.FileHandler(os.path.join(self.log_directory, 'project.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler_stream.setFormatter(formatter)
            handler_file.setFormatter(formatter)
            logger.addHandler(handler_stream)
            logger.addHandler(handler_file)
            return logger
        except Exception as e:
            print(f"Error setting up logger: {e}")
            raise RuntimeError(f"Error setting up logger: {e}")

    def log_info(self, message):
        """Helper function to log info messages."""
        try:
            self.logger.info(message)
        except Exception as e:
            print(f"Error logging info message: {e}")
            raise RuntimeError(f"Error logging info message: {e}")

    def log_error(self, message):
        """Helper function to log error messages."""
        try:
            self.logger.error(message)
        except Exception as e:
            print(f"Error logging error message: {e}")
            raise RuntimeError(f"Error logging error message: {e}")

    def log_warning(self, message):
        """Helper function to log warning messages."""
        try:
            self.logger.warning(message)
        except Exception as e:
            print(f"Error logging warning message: {e}")
            raise RuntimeError(f"Error logging warning message: {e}")