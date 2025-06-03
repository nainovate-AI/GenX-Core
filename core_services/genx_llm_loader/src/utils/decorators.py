from functools import wraps
from src.telemetry.opentelemetry import GenxTelemetry
import logging

logger = logging.getLogger(__name__)

def log_and_trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        telemetry = GenxTelemetry()
        with telemetry.start_span(func.__name__):
            logger.info(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
    return wrapper