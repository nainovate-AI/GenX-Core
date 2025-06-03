import grpc
from concurrent import futures
from src.proto import llm_loader_pb2, llm_loader_pb2_grpc
from src.services.model_manager import GenxModelManager
from src.utils.exceptions import GenxModelLoadError, GenxModelNotFoundError
import logging

logger = logging.getLogger(__name__)

class GenxLLMLoaderService(llm_loader_pb2_grpc.LLMLoaderServicer):
    def __init__(self):
        self.model_manager = GenxModelManager()

    def LoadModel(self, request, context):
        try:
            success = self.model_manager.load_model(
                controller=request.controller,
                model_id=request.model_id,
                quantization_type=request.quantization_type,
                parameters=dict(request.parameters),
                device=request.device,
            )
            return llm_loader_pb2.LoadModelResponse(
                success=success, message="Model loaded successfully", model_id=request.model_id
            )
        except GenxModelLoadError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return llm_loader_pb2.LoadModelResponse(success=False, message=str(e))

    def UnloadModel(self, request, context):
        try:
            success = self.model_manager.unload_model(request.model_id)
            return llm_loader_pb2.UnloadModelResponse(
                success=success, message="Model unloaded successfully"
            )
        except GenxModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return llm_loader_pb2.UnloadModelResponse(success=False, message=str(e))

    def GetModelDetails(self, request, context):
        try:
            details = self.model_manager.get_model_details(request.model_id)
            return llm_loader_pb2.GetModelDetailsResponse(
                success=True,
                model_id=request.model_id,
                controller=details.get("controller", ""),
                device=details.get("device", ""),
                parameters=details.get("parameters", {}),
                message="Model details retrieved successfully",
            )
        except GenxModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return llm_loader_pb2.GetModelDetailsResponse(success=False, message=str(e))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llm_loader_pb2_grpc.add_LLMLoaderServicer_to_server(GenxLLMLoaderService(), server)
    server.add_insecure_port("[::]:50051")
    logger.info("Starting gRPC server on port 50051")
    server.start()
    server.wait_for_termination()