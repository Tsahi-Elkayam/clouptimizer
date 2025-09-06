from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"


@dataclass
class CloudCredentials:
    """Base class for cloud credentials"""
    provider: CloudProvider
    profile: Optional[str] = None
    region: Optional[str] = None


class BaseCloudProvider(ABC):
    """Abstract base class for cloud provider implementations"""
    
    def __init__(self, credentials: CloudCredentials):
        self.credentials = credentials
        self.client = None
        self._authenticated = False
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the cloud provider"""
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate the provided credentials"""
        pass
    
    @abstractmethod
    def get_regions(self) -> List[str]:
        """Get list of available regions"""
        pass
    
    @abstractmethod
    def get_services(self) -> List[str]:
        """Get list of available services"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the cloud provider"""
        pass
    
    @property
    def is_authenticated(self) -> bool:
        """Check if provider is authenticated"""
        return self._authenticated
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": self.credentials.provider.value,
            "authenticated": self._authenticated,
            "region": self.credentials.region,
            "profile": self.credentials.profile
        }