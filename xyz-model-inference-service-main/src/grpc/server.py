"""Start and manage the gRPC server on port 9090."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import grpc
from grpc_reflection.v1alpha import reflection

from src.grpc import inference_pb2, inference_pb2_grpc
from src.grpc.servicer import ModelInferenceServicer

if TYPE_CHECKING:
    from src.services.inference_service import InferenceService
    from src.adapters.triton_client import TritonClient

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9090
_MAX_WORKERS = 10
_MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50 MB


class GrpcServer:
    """Manages lifecycle of the async gRPC server."""

    def __init__(self, port: int = _DEFAULT_PORT):
        self._port = port
        self._server: Optional[grpc.aio.Server] = None

    async def start(
        self,
        inference_service: "InferenceService",
        triton_client: "TritonClient",
    ) -> None:
        servicer = ModelInferenceServicer(inference_service, triton_client)

        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
            ]
        )

        inference_pb2_grpc.add_ModelInferenceServiceServicer_to_server(
            servicer, self._server
        )

        # Register gRPC reflection so grpcurl/clients can discover the service
        service_names = (
            inference_pb2.DESCRIPTOR.services_by_name["ModelInferenceService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, self._server)

        listen_addr = f"[::]:{self._port}"
        self._server.add_insecure_port(listen_addr)
        await self._server.start()
        logger.info("gRPC server listening on %s", listen_addr)

    async def stop(self, grace_s: float = 5.0) -> None:
        if self._server:
            await self._server.stop(grace_s)
            logger.info("gRPC server stopped")
