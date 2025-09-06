from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    ANALYTICS = "analytics"
    AI_ML = "ai_ml"
    SECURITY = "security"
    MONITORING = "monitoring"
    OTHER = "other"


class ResourceStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    IDLE = "idle"
    DELETED = "deleted"
    UNKNOWN = "unknown"


@dataclass
class ResourceMetadata:
    """Metadata for cloud resources"""
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None


@dataclass
class ResourceCost:
    """Cost information for a resource"""
    hourly_cost: float = 0.0
    monthly_cost: float = 0.0
    annual_cost: float = 0.0
    currency: str = "USD"
    last_updated: Optional[datetime] = None


@dataclass
class ResourceMetrics:
    """Performance metrics for a resource"""
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    network_in: Optional[float] = None
    network_out: Optional[float] = None
    disk_read: Optional[float] = None
    disk_write: Optional[float] = None
    request_count: Optional[int] = None
    error_rate: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseResource(ABC):
    """Base class for all cloud resources"""
    
    resource_id: str
    name: str
    resource_type: ResourceType
    provider: str
    status: ResourceStatus = ResourceStatus.UNKNOWN
    metadata: ResourceMetadata = field(default_factory=ResourceMetadata)
    cost: ResourceCost = field(default_factory=ResourceCost)
    metrics: ResourceMetrics = field(default_factory=ResourceMetrics)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_monthly_cost(self) -> float:
        """Get estimated monthly cost"""
        return self.cost.monthly_cost
    
    def get_annual_cost(self) -> float:
        """Get estimated annual cost"""
        return self.cost.annual_cost or (self.cost.monthly_cost * 12)
    
    def is_idle(self) -> bool:
        """Check if resource is idle based on metrics"""
        if self.status == ResourceStatus.STOPPED:
            return True
        
        if self.metrics.cpu_utilization is not None and self.metrics.cpu_utilization < 5:
            return True
        
        return False
    
    def get_utilization_score(self) -> float:
        """Calculate overall utilization score (0-100)"""
        scores = []
        
        if self.metrics.cpu_utilization is not None:
            scores.append(self.metrics.cpu_utilization)
        
        if self.metrics.memory_utilization is not None:
            scores.append(self.metrics.memory_utilization)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary"""
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "type": self.resource_type.value,
            "provider": self.provider,
            "status": self.status.value,
            "region": self.metadata.region,
            "tags": self.metadata.tags,
            "monthly_cost": self.cost.monthly_cost,
            "utilization_score": self.get_utilization_score(),
            "is_idle": self.is_idle()
        }