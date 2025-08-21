"""API routes for VSR API."""

from vsr_api.routes.uploads import router as uploads_router
from vsr_api.routes.jobs import router as jobs_router

__all__ = ["uploads_router", "jobs_router"]
