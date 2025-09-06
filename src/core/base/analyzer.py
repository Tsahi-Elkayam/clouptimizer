from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .resource import BaseResource


class OptimizationType(Enum):
    RIGHTSIZING = "rightsizing"
    IDLE_RESOURCE = "idle_resource"
    UNUSED_RESOURCE = "unused_resource"
    RESERVED_INSTANCE = "reserved_instance"
    SAVINGS_PLAN = "savings_plan"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    SCHEDULING = "scheduling"
    MIGRATION = "migration"
    CONSOLIDATION = "consolidation"
    OTHER = "other"


class OptimizationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OptimizationComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class Optimization:
    """Represents a single optimization opportunity"""
    
    optimization_id: str
    resource: BaseResource
    type: OptimizationType
    title: str
    description: str
    current_cost: float
    optimized_cost: float
    savings: float
    savings_percentage: float
    priority: OptimizationPriority
    complexity: OptimizationComplexity
    confidence_score: float  # 0-100
    implementation_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_annual_savings(self) -> float:
        """Calculate annual savings"""
        return self.savings * 12
    
    def get_roi_days(self) -> float:
        """Calculate ROI in days based on effort"""
        if self.estimated_effort_hours == 0:
            return 0
        
        # Assume $150/hour for implementation cost
        implementation_cost = self.estimated_effort_hours * 150
        daily_savings = self.savings * 12 / 365
        
        if daily_savings == 0:
            return float('inf')
        
        return implementation_cost / daily_savings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimization_id": self.optimization_id,
            "resource_id": self.resource.resource_id,
            "resource_name": self.resource.name,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "current_cost": self.current_cost,
            "optimized_cost": self.optimized_cost,
            "monthly_savings": self.savings,
            "annual_savings": self.get_annual_savings(),
            "savings_percentage": self.savings_percentage,
            "priority": self.priority.value,
            "complexity": self.complexity.value,
            "confidence_score": self.confidence_score,
            "roi_days": self.get_roi_days(),
            "implementation_steps": self.implementation_steps,
            "risks": self.risks
        }


@dataclass
class AnalysisResult:
    """Results from an analysis run"""
    
    analyzer_name: str
    timestamp: datetime
    resources_analyzed: int
    optimizations: List[Optimization]
    total_monthly_savings: float
    total_annual_savings: float
    execution_time_seconds: float
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_optimizations_by_type(self) -> Dict[OptimizationType, List[Optimization]]:
        """Group optimizations by type"""
        grouped = {}
        for opt in self.optimizations:
            if opt.type not in grouped:
                grouped[opt.type] = []
            grouped[opt.type].append(opt)
        return grouped
    
    def get_optimizations_by_priority(self) -> Dict[OptimizationPriority, List[Optimization]]:
        """Group optimizations by priority"""
        grouped = {}
        for opt in self.optimizations:
            if opt.priority not in grouped:
                grouped[opt.priority] = []
            grouped[opt.priority].append(opt)
        return grouped
    
    def get_top_optimizations(self, n: int = 10) -> List[Optimization]:
        """Get top N optimizations by savings"""
        return sorted(self.optimizations, key=lambda x: x.savings, reverse=True)[:n]


class BaseAnalyzer(ABC):
    """Abstract base class for resource analyzers"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.optimizations: List[Optimization] = []
    
    @abstractmethod
    def analyze(self, resources: List[BaseResource]) -> AnalysisResult:
        """Analyze resources and return optimization opportunities"""
        pass
    
    @abstractmethod
    def analyze_resource(self, resource: BaseResource) -> Optional[Optimization]:
        """Analyze a single resource"""
        pass
    
    def calculate_savings(self, current_cost: float, optimized_cost: float) -> Dict[str, float]:
        """Calculate savings metrics"""
        savings = current_cost - optimized_cost
        savings_percentage = (savings / current_cost * 100) if current_cost > 0 else 0
        
        return {
            "savings": savings,
            "savings_percentage": savings_percentage,
            "annual_savings": savings * 12
        }
    
    def determine_priority(self, savings: float, confidence: float) -> OptimizationPriority:
        """Determine optimization priority based on savings and confidence"""
        if savings > 1000 and confidence > 80:
            return OptimizationPriority.CRITICAL
        elif savings > 500 or confidence > 70:
            return OptimizationPriority.HIGH
        elif savings > 100 or confidence > 50:
            return OptimizationPriority.MEDIUM
        else:
            return OptimizationPriority.LOW
    
    def create_optimization(
        self,
        resource: BaseResource,
        opt_type: OptimizationType,
        title: str,
        description: str,
        current_cost: float,
        optimized_cost: float,
        **kwargs
    ) -> Optimization:
        """Create an optimization object"""
        
        savings_data = self.calculate_savings(current_cost, optimized_cost)
        
        return Optimization(
            optimization_id=f"{self.name}_{resource.resource_id}_{datetime.now().timestamp()}",
            resource=resource,
            type=opt_type,
            title=title,
            description=description,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings=savings_data["savings"],
            savings_percentage=savings_data["savings_percentage"],
            priority=kwargs.get("priority", self.determine_priority(
                savings_data["savings"],
                kwargs.get("confidence_score", 50)
            )),
            complexity=kwargs.get("complexity", OptimizationComplexity.MODERATE),
            confidence_score=kwargs.get("confidence_score", 50),
            implementation_steps=kwargs.get("implementation_steps", []),
            risks=kwargs.get("risks", []),
            prerequisites=kwargs.get("prerequisites", []),
            estimated_effort_hours=kwargs.get("estimated_effort_hours", 0),
            metadata=kwargs.get("metadata", {})
        )