from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .resource import BaseResource


@dataclass
class CollectorConfig:
    """Configuration for resource collectors"""
    regions: List[str] = None
    resource_types: List[str] = None
    tags_filter: Dict[str, str] = None
    max_workers: int = 10
    timeout: int = 300
    include_metrics: bool = True
    include_costs: bool = True


class BaseCollector(ABC):
    """Abstract base class for resource collectors"""
    
    def __init__(self, provider, config: CollectorConfig = None):
        self.provider = provider
        self.config = config or CollectorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.resources: List[BaseResource] = []
        self.errors: List[Dict[str, Any]] = []
    
    @abstractmethod
    def collect(self) -> List[BaseResource]:
        """Collect resources from the cloud provider"""
        pass
    
    @abstractmethod
    def get_resource_details(self, resource_id: str) -> BaseResource:
        """Get detailed information about a specific resource"""
        pass
    
    @abstractmethod
    def get_resource_metrics(self, resource: BaseResource) -> Dict[str, Any]:
        """Get metrics for a resource"""
        pass
    
    @abstractmethod
    def get_resource_cost(self, resource: BaseResource) -> Dict[str, float]:
        """Get cost information for a resource"""
        pass
    
    def collect_parallel(self, collect_funcs: List[callable]) -> List[BaseResource]:
        """Execute collection functions in parallel"""
        all_resources = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(func): func.__name__ for func in collect_funcs}
            
            for future in as_completed(futures):
                func_name = futures[future]
                try:
                    resources = future.result(timeout=self.config.timeout)
                    all_resources.extend(resources)
                    self.logger.info(f"Collected {len(resources)} resources from {func_name}")
                except Exception as e:
                    self.logger.error(f"Error collecting from {func_name}: {str(e)}")
                    self.errors.append({
                        "function": func_name,
                        "error": str(e),
                        "timestamp": datetime.now()
                    })
        
        return all_resources
    
    def filter_resources(self, resources: List[BaseResource]) -> List[BaseResource]:
        """Filter resources based on configuration"""
        filtered = resources
        
        if self.config.tags_filter:
            filtered = [
                r for r in filtered
                if all(
                    r.metadata.tags.get(k) == v
                    for k, v in self.config.tags_filter.items()
                )
            ]
        
        if self.config.regions:
            filtered = [
                r for r in filtered
                if r.metadata.region in self.config.regions
            ]
        
        return filtered
    
    def enrich_with_metrics(self, resources: List[BaseResource]) -> None:
        """Enrich resources with metrics data"""
        if not self.config.include_metrics:
            return
        
        for resource in resources:
            try:
                metrics = self.get_resource_metrics(resource)
                if metrics:
                    resource.metrics.cpu_utilization = metrics.get('cpu_utilization')
                    resource.metrics.memory_utilization = metrics.get('memory_utilization')
                    resource.metrics.network_in = metrics.get('network_in')
                    resource.metrics.network_out = metrics.get('network_out')
            except Exception as e:
                self.logger.warning(f"Failed to get metrics for {resource.resource_id}: {e}")
    
    def enrich_with_costs(self, resources: List[BaseResource]) -> None:
        """Enrich resources with cost data"""
        if not self.config.include_costs:
            return
        
        for resource in resources:
            try:
                costs = self.get_resource_cost(resource)
                if costs:
                    resource.cost.hourly_cost = costs.get('hourly', 0)
                    resource.cost.monthly_cost = costs.get('monthly', 0)
                    resource.cost.annual_cost = costs.get('annual', 0)
            except Exception as e:
                self.logger.warning(f"Failed to get costs for {resource.resource_id}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get collection summary"""
        return {
            "total_resources": len(self.resources),
            "total_errors": len(self.errors),
            "resources_by_type": self._count_by_type(),
            "resources_by_region": self._count_by_region(),
            "total_monthly_cost": sum(r.cost.monthly_cost for r in self.resources),
            "errors": self.errors
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count resources by type"""
        counts = {}
        for resource in self.resources:
            type_name = resource.resource_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def _count_by_region(self) -> Dict[str, int]:
        """Count resources by region"""
        counts = {}
        for resource in self.resources:
            region = resource.metadata.region or "unknown"
            counts[region] = counts.get(region, 0) + 1
        return counts