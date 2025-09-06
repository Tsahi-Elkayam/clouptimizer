"""FastAPI application for Clouptimizer API"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Optional, Dict, Any
import uvicorn

from ..core.config import get_settings, Settings
from ..core.logging import setup_logging, get_logger
from ..core.monitoring import MetricsCollector, SystemMonitor, HealthChecker
from ..core.security import RateLimiter, TokenManager, APIKeyManager, SecurityHeaders
from ..core.exceptions import CloudOptimizerError, RateLimitError, AuthenticationError

logger = get_logger(__name__)

# Global instances
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)
health_checker = HealthChecker()
rate_limiter = RateLimiter()
token_manager = TokenManager()
api_key_manager = APIKeyManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Clouptimizer API")
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.logging.level,
        structured=settings.logging.structured,
        console=settings.logging.console,
        log_file=settings.logging.file
    )
    
    # Start monitoring
    metrics_collector.start()
    system_monitor.start(interval=settings.monitoring.export_interval)
    
    # Register health checks
    register_health_checks()
    
    logger.info("Clouptimizer API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Clouptimizer API")
    system_monitor.stop()
    metrics_collector.stop()
    logger.info("Clouptimizer API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Clouptimizer API",
    description="Multi-cloud cost optimization API",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure based on settings


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)
    SecurityHeaders.apply(response.headers)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "")
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else "unknown"
    })
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {duration:.3f}s", extra={
        "request_id": request_id,
        "status_code": response.status_code,
        "duration": duration
    })
    
    # Record metrics
    metrics_collector.record_histogram("api.request.duration", duration, {
        "method": request.method,
        "path": request.url.path,
        "status": str(response.status_code)
    })
    
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting"""
    settings = get_settings()
    if not settings.security.rate_limit_per_minute:
        return await call_next(request)
    
    # Get client identifier
    client = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not rate_limiter.is_allowed(client):
        reset_time = rate_limiter.get_reset_time(client)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "reset_in": reset_time
            },
            headers={
                "X-RateLimit-Limit": str(settings.security.rate_limit_per_minute),
                "X-RateLimit-Reset": str(reset_time)
            }
        )
    
    return await call_next(request)


@app.exception_handler(CloudOptimizerError)
async def cloud_optimizer_exception_handler(request: Request, exc: CloudOptimizerError):
    """Handle application exceptions"""
    logger.error(f"Application error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc)}
    )


@app.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
    """Handle rate limit exceptions"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": str(exc)}
    )


@app.exception_handler(AuthenticationError)
async def auth_exception_handler(request: Request, exc: AuthenticationError):
    """Handle authentication exceptions"""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": str(exc)}
    )


# Health check endpoints
@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    health_status = health_checker.check_health()
    
    if not health_status.healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not ready",
                "checks": health_status.checks,
                "message": health_status.message
            }
        )
    
    return {
        "status": "ready",
        "checks": health_status.checks
    }


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with all subsystems"""
    health_status = health_checker.check_health()
    system_stats = system_monitor.get_system_stats()
    
    return {
        "status": "healthy" if health_status.healthy else "unhealthy",
        "timestamp": time.time(),
        "checks": health_status.checks,
        "system": system_stats,
        "details": health_status.details
    }


# Metrics endpoints
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    metrics_data = metrics_collector.export_prometheus()
    return JSONResponse(
        content=metrics_data,
        media_type="text/plain"
    )


@app.get("/metrics/json")
async def metrics_json():
    """JSON metrics endpoint"""
    return {
        "counters": dict(metrics_collector.counters),
        "gauges": dict(metrics_collector.gauges),
        "histograms": {
            k: metrics_collector.get_histogram_stats(k)
            for k in metrics_collector.histograms.keys()
        }
    }


# API info endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/info")
async def info():
    """API information"""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "providers": settings.get_enabled_providers(),
        "features": {
            "logging": settings.logging.structured,
            "monitoring": settings.monitoring.enabled,
            "caching": settings.cache.enabled,
            "notifications": settings.notifications.enabled,
            "database": settings.database.enabled
        }
    }


def register_health_checks():
    """Register application health checks"""
    
    def check_database():
        """Check database connection"""
        settings = get_settings()
        if not settings.database.enabled:
            return True
        # TODO: Implement actual database check
        return True
    
    def check_cache():
        """Check cache connection"""
        settings = get_settings()
        if not settings.cache.enabled:
            return True
        # TODO: Implement actual cache check
        return True
    
    def check_providers():
        """Check cloud provider connections"""
        # TODO: Implement actual provider checks
        return True
    
    health_checker.register_check("database", check_database)
    health_checker.register_check("cache", check_cache)
    health_checker.register_check("providers", check_providers)


def create_app() -> FastAPI:
    """Create and configure FastAPI app"""
    return app


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.logging.level.lower()
    )