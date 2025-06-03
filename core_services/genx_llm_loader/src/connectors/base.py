from abc import ABC, abstractmethod
from typing import Dict, Optional

class GenxBaseConnector(ABC):
    @abstractmethod
    def load_model(
        self, model_id: str, quantization_type: Optional[str], parameters: Dict[str, str], device: str
    ) -> bool:
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        pass

    @abstractmethod
    def get_model_details(self, model_id: str) -> Dict[str, str]:
        pass