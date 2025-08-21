"""Main FastAPI application module."""

import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vsr_api.config import get_settings
from vsr_api.routes import uploads_router, jobs_router
from vsr_api.routes.submit import router as submit_router
from vsr_api.routes.health import router as health_router
from vsr_api.routes.metrics import router as metrics_router
from vsr_api.middleware.auth import APIKeyAuthMiddleware
from vsr_api.middleware.rate_limit import rate_limit_middleware, cleanup_rate_limiter
from vsr_api.middleware.metrics import MetricsMiddleware
from vsr_api.middleware.quotas import QuotasMiddleware
from vsr_api.middleware.security import SecurityMiddleware
from vsr_api.docs.openapi_config import configure_openapi_docs
from vsr_shared.db import mongodb_lifespan
from vsr_shared.logging import bind_request_id, clear_request_id, get_logger, setup_logging
from vsr_shared.queue.integration import create_rabbitmq_lifespan

# Set up logging
setup_logging(level="INFO", json_format=True)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    
    Args:
        app: The FastAPI application instance
    """
    # Startup logic
    settings = get_settings()
    app.state.settings = settings
    
    logger.info("Starting API", version=app.version)
    
    # Initialize RabbitMQ lifespan manager
    rabbitmq_lifespan = create_rabbitmq_lifespan(
        rabbitmq_url=settings.rabbitmq_url
    )
    
    # Start RabbitMQ connection
    await rabbitmq_lifespan.startup()
    
    # Start background tasks
    import asyncio
    cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    
    try:
        # MongoDB connection
        async with mongodb_lifespan():
            logger.info("MongoDB and RabbitMQ connections established")
            yield
    finally:
        # Cancel background tasks
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        
        # Shutdown RabbitMQ
        await rabbitmq_lifespan.shutdown()
    
    # Shutdown logic
    logger.info("Shutting down API")


app = FastAPI(
    title="Video Subtitle Removal API",
    description="API for removing hardcoded subtitles from videos using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure comprehensive OpenAPI documentation
configure_openapi_docs(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add observability and security middleware (order matters)
app.add_middleware(SecurityMiddleware)  # Security first
app.add_middleware(MetricsMiddleware)   # Metrics collection
app.add_middleware(QuotasMiddleware)    # Quotas and limits

# Add custom middleware (order matters - auth before rate limiting)
# app.add_middleware(APIKeyAuthMiddleware)  # Disabled for local development
# app.middleware("http")(rate_limit_middleware)  # Disabled for local development


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware to add request ID to each request for tracing.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or route handler
        
    Returns:
        Response from the next middleware or route handler
    """
    request_id = str(uuid.uuid4())
    bind_request_id(request_id)
    
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_id()


@app.get("/healthz", tags=["Health"])
async def health_check() -> JSONResponse:
    """
    Health check endpoint.
    
    Returns:
        JSONResponse: Status of the API
    """
    logger.debug("Health check requested")
    return JSONResponse(content={"status": "ok"})


# Include routers
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(uploads_router)
app.include_router(jobs_router)
app.include_router(submit_router)


if __name__ == "__main__":
    uvicorn.run(
        "vsr_api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
