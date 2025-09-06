"""Custom exceptions for Clouptimizer"""

class CloudOptimizerError(Exception):
    """Base exception for all Clouptimizer errors"""
    pass


class AuthenticationError(CloudOptimizerError):
    """Raised when authentication fails"""
    pass


class ConfigurationError(CloudOptimizerError):
    """Raised when configuration is invalid"""
    pass


class ResourceAccessError(CloudOptimizerError):
    """Raised when resource access fails"""
    pass


class ValidationError(CloudOptimizerError):
    """Raised when input validation fails"""
    pass


class RateLimitError(CloudOptimizerError):
    """Raised when rate limit is exceeded"""
    pass


class DataCollectionError(CloudOptimizerError):
    """Raised when data collection fails"""
    pass


class AnalysisError(CloudOptimizerError):
    """Raised when analysis fails"""
    pass


class OptimizationError(CloudOptimizerError):
    """Raised when optimization fails"""
    pass


class ReportGenerationError(CloudOptimizerError):
    """Raised when report generation fails"""
    pass


class NetworkError(CloudOptimizerError):
    """Raised when network operations fail"""
    pass


class TimeoutError(CloudOptimizerError):
    """Raised when operation times out"""
    pass


class ProviderError(CloudOptimizerError):
    """Base exception for provider-specific errors"""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AWSError(ProviderError):
    """AWS-specific errors"""
    def __init__(self, message: str):
        super().__init__("AWS", message)


class AzureError(ProviderError):
    """Azure-specific errors"""
    def __init__(self, message: str):
        super().__init__("Azure", message)


class GCPError(ProviderError):
    """GCP-specific errors"""
    def __init__(self, message: str):
        super().__init__("GCP", message)