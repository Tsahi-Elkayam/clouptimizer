from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .analyzer import Optimization, OptimizationType


class ActionType(Enum):
    RESIZE = "resize"
    TERMINATE = "terminate"
    STOP = "stop"
    SCHEDULE = "schedule"
    MIGRATE = "migrate"
    PURCHASE_RI = "purchase_ri"
    MODIFY_RI = "modify_ri"
    CHANGE_STORAGE_CLASS = "change_storage_class"
    ENABLE_FEATURE = "enable_feature"
    DISABLE_FEATURE = "disable_feature"
    TAG = "tag"
    OTHER = "other"


class ActionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class OptimizationAction:
    """Represents an action to implement an optimization"""
    
    action_id: str
    optimization: Optimization
    action_type: ActionType
    description: str
    commands: List[str] = field(default_factory=list)
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    terraform_code: Optional[str] = None
    estimated_duration_minutes: float = 0
    requires_downtime: bool = False
    requires_approval: bool = True
    rollback_commands: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    """Execution plan for optimizations"""
    
    plan_id: str
    name: str
    description: str
    optimizations: List[Optimization]
    actions: List[OptimizationAction]
    total_savings: float
    total_effort_hours: float
    created_at: datetime
    phases: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_phase_1_actions(self) -> List[OptimizationAction]:
        """Get quick win actions (low risk, high impact)"""
        return [
            action for action in self.actions
            if action.optimization.complexity.value == "simple"
            and action.optimization.confidence_score > 80
        ]
    
    def get_actions_by_type(self) -> Dict[ActionType, List[OptimizationAction]]:
        """Group actions by type"""
        grouped = {}
        for action in self.actions:
            if action.action_type not in grouped:
                grouped[action.action_type] = []
            grouped[action.action_type].append(action)
        return grouped
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "total_optimizations": len(self.optimizations),
            "total_actions": len(self.actions),
            "total_monthly_savings": self.total_savings,
            "total_annual_savings": self.total_savings * 12,
            "total_effort_hours": self.total_effort_hours,
            "created_at": self.created_at.isoformat(),
            "phases": self.phases
        }


@dataclass
class ExecutionResult:
    """Result of executing an optimization action"""
    
    action_id: str
    status: ActionStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_minutes: float
    success: bool
    message: str
    output: Optional[str] = None
    error: Optional[str] = None
    rollback_performed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Abstract base class for optimization executors"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.execution_results: List[ExecutionResult] = []
    
    @abstractmethod
    def create_action(self, optimization: Optimization) -> OptimizationAction:
        """Create an action for an optimization"""
        pass
    
    @abstractmethod
    def execute_action(self, action: OptimizationAction) -> ExecutionResult:
        """Execute an optimization action"""
        pass
    
    @abstractmethod
    def validate_action(self, action: OptimizationAction) -> bool:
        """Validate if an action can be executed"""
        pass
    
    @abstractmethod
    def rollback_action(self, action: OptimizationAction, result: ExecutionResult) -> bool:
        """Rollback an action if it fails"""
        pass
    
    def create_optimization_plan(
        self,
        optimizations: List[Optimization],
        name: str = "Optimization Plan"
    ) -> OptimizationPlan:
        """Create an execution plan from optimizations"""
        
        actions = []
        for opt in optimizations:
            action = self.create_action(opt)
            if action:
                actions.append(action)
        
        # Group into phases
        phases = [
            {
                "name": "Phase 1: Quick Wins",
                "description": "Low risk, high impact optimizations",
                "actions": [a.action_id for a in actions if a.optimization.complexity.value == "simple"],
                "estimated_savings": sum(
                    a.optimization.savings for a in actions
                    if a.optimization.complexity.value == "simple"
                )
            },
            {
                "name": "Phase 2: Standard Optimizations",
                "description": "Moderate complexity optimizations",
                "actions": [a.action_id for a in actions if a.optimization.complexity.value == "moderate"],
                "estimated_savings": sum(
                    a.optimization.savings for a in actions
                    if a.optimization.complexity.value == "moderate"
                )
            },
            {
                "name": "Phase 3: Complex Transformations",
                "description": "High complexity, high value optimizations",
                "actions": [a.action_id for a in actions if a.optimization.complexity.value == "complex"],
                "estimated_savings": sum(
                    a.optimization.savings for a in actions
                    if a.optimization.complexity.value == "complex"
                )
            }
        ]
        
        return OptimizationPlan(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=f"Optimization plan with {len(optimizations)} optimizations",
            optimizations=optimizations,
            actions=actions,
            total_savings=sum(opt.savings for opt in optimizations),
            total_effort_hours=sum(opt.estimated_effort_hours for opt in optimizations),
            created_at=datetime.now(),
            phases=phases
        )
    
    def execute_plan(self, plan: OptimizationPlan, phase: int = None) -> List[ExecutionResult]:
        """Execute an optimization plan"""
        
        results = []
        
        # Get actions to execute
        if phase is not None and 0 <= phase < len(plan.phases):
            action_ids = plan.phases[phase]["actions"]
            actions = [a for a in plan.actions if a.action_id in action_ids]
        else:
            actions = plan.actions
        
        for action in actions:
            if self.validate_action(action):
                result = self.execute_action(action)
                results.append(result)
                
                if not result.success and not self.dry_run:
                    # Attempt rollback
                    self.rollback_action(action, result)
            else:
                results.append(ExecutionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SKIPPED,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    duration_minutes=0,
                    success=False,
                    message="Action validation failed"
                ))
        
        self.execution_results.extend(results)
        return results