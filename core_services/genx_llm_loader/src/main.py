import logging
from src.services.grpc_service import serve
from src.telemetry.opentelemetry import GenxTelemetry

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize telemetry
    telemetry = GenxTelemetry()
    
    logger.info("Starting Genx LLM Loader Microservice")
    try:
        serve()
    except Exception as e:
        logger.error(f"Failed to start microservice: {str(e)}")
        raise

if __name__ == "__main__":
    main()