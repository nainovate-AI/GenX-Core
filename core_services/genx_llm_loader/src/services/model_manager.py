from src.connectors.tgi import GenxTGIConnector
from src.connectors.vllm import GenxVLLMConnector
from src.utils.decorators import log_and_trace
from src.utils.exceptions import GenxModelLoadError, GenxModelNotFoundError
from typing import Dict, Optional

class GenxModelManager:
    def __init__(self):
        self.connectors = {
            "tgi": GenxTGIConnector(),
            "vllm": GenxVLLMConnector(),
        }

    @log_and_trace
    def load_model(
        self, controller: str, model_id: str, quantization_type: Optional[str], parameters: Dict[str, str], device: str
    ) -> bool:
        if controller not in self.connectors:
            raise GenxModelLoadError(f"Unsupported controller: {controller}")
        try:
            return self.connectors[controller].load_model(
                model_id, quantization_type, parameters, device
            )
        except Exception as e:
            raise GenxModelLoadError(f"Failed to load model {model_id}: {str(e)}")

    @log_and_trace
    def unload_model(self, model_id: str) -> bool:
        for connector in self.connectors.values():
            if connector.get_model_details(model_id):
                return connector.unload_model(model_id)
        raise GenxModelNotFoundError(f"Model {model_id} not found")

    @log_and_trace
    def get_model_details(self, model_id: str) -> Dict[str, str]:
        for connector in self.connectors.values():
            details = connector.get_model_details(model_id)
            if details:
                return details
        raise GenxModelNotFoundError(f"Model {model_id} not found")