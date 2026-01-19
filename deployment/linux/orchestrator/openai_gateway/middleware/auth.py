"""
Authentication middleware
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import BACKEND_API_KEY


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication"""

    def __init__(self, app, exclude_paths=None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for certain paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Authorization header required",
                        "type": "authentication_error",
                        "param": None,
                        "code": None
                    }
                }
            )

        # Extract Bearer token
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid authorization format",
                        "type": "authentication_error",
                        "param": None,
                        "code": None
                    }
                }
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token (simple check against configured key)
        if token != BACKEND_API_KEY:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                        "param": None,
                        "code": None
                    }
                }
            )

        # Proceed with request
        response = await call_next(request)
        return response