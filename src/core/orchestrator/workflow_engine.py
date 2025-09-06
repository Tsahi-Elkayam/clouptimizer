import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..base import (
    BaseCloudProvider, BaseCollector, BaseAnalyzer,
    BaseOptimizer, BaseResource, AnalysisResult,
    OptimizationPlan
)


@dataclass
class WorkflowStep:
    """Represents a step in the workflow"""
    name: str
    type: str  # 'collect', 'analyze', 'optimize', 'report'
    provider: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    success: bool
    resources_collected: List[BaseResource]
    analysis_results: List[AnalysisResult]
    optimization_plan: Optional[OptimizationPlan]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "total_resources": len(self.resources_collected),
            "total_optimizations": sum(len(r.optimizations) for r in self.analysis_results),
            "total_savings": sum(r.total_monthly_savings for r in self.analysis_results),
            "errors": self.errors
        }


class WorkflowEngine:
    """Orchestrates the optimization workflow"""
    
    def __init__(self, providers: Dict[str, BaseCloudProvider] = None):
        self.providers = providers or {}
        self.collectors: Dict[str, BaseCollector] = {}
        self.analyzers: List[BaseAnalyzer] = []
        self.optimizer: Optional[BaseOptimizer] = None
        self.logger = logging.getLogger(__name__)
        self.current_workflow: Optional[WorkflowResult] = None
    
    def register_provider(self, name: str, provider: BaseCloudProvider):
        """Register a cloud provider"""
        self.providers[name] = provider
        self.logger.info(f"Registered provider: {name}")
    
    def register_collector(self, name: str, collector: BaseCollector):
        """Register a resource collector"""
        self.collectors[name] = collector
        self.logger.info(f"Registered collector: {name}")
    
    def register_analyzer(self, analyzer: BaseAnalyzer):
        """Register an analyzer"""
        self.analyzers.append(analyzer)
        self.logger.info(f"Registered analyzer: {analyzer.name}")
    
    def register_optimizer(self, optimizer: BaseOptimizer):
        """Register an optimizer"""
        self.optimizer = optimizer
        self.logger.info("Registered optimizer")
    
    def create_workflow(self, steps: List[WorkflowStep]) -> str:
        """Create a new workflow"""
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_workflow = WorkflowResult(
            workflow_id=workflow_id,
            started_at=datetime.now(),
            completed_at=None,
            success=False,
            resources_collected=[],
            analysis_results=[],
            optimization_plan=None,
            errors=[],
            metadata={"steps": [s.name for s in steps]}
        )
        
        return workflow_id
    
    def execute_workflow(self, steps: List[WorkflowStep]) -> WorkflowResult:
        """Execute a complete workflow"""
        
        workflow_id = self.create_workflow(steps)
        self.logger.info(f"Starting workflow: {workflow_id}")
        
        try:
            # Group steps by type
            collect_steps = [s for s in steps if s.type == 'collect']
            analyze_steps = [s for s in steps if s.type == 'analyze']
            optimize_steps = [s for s in steps if s.type == 'optimize']
            
            # Phase 1: Collection
            if collect_steps:
                resources = self._execute_collection(collect_steps)
                self.current_workflow.resources_collected = resources
                self.logger.info(f"Collected {len(resources)} resources")
            
            # Phase 2: Analysis
            if analyze_steps and self.current_workflow.resources_collected:
                analysis_results = self._execute_analysis(
                    self.current_workflow.resources_collected
                )
                self.current_workflow.analysis_results = analysis_results
                self.logger.info(f"Generated {sum(len(r.optimizations) for r in analysis_results)} optimizations")
            
            # Phase 3: Optimization Planning
            if optimize_steps and self.current_workflow.analysis_results:
                optimization_plan = self._execute_optimization_planning(
                    self.current_workflow.analysis_results
                )
                self.current_workflow.optimization_plan = optimization_plan
                self.logger.info(f"Created optimization plan with {len(optimization_plan.actions)} actions")
            
            self.current_workflow.success = True
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            self.current_workflow.errors.append({
                "phase": "workflow",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.current_workflow.success = False
        
        finally:
            self.current_workflow.completed_at = datetime.now()
        
        return self.current_workflow
    
    def _execute_collection(self, steps: List[WorkflowStep]) -> List[BaseResource]:
        """Execute collection phase"""
        all_resources = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for step in steps:
                if step.provider in self.collectors:
                    collector = self.collectors[step.provider]
                    future = executor.submit(collector.collect)
                    futures[future] = step.name
            
            for future in as_completed(futures):
                step_name = futures[future]
                try:
                    resources = future.result(timeout=300)
                    all_resources.extend(resources)
                    self.logger.info(f"Step {step_name}: collected {len(resources)} resources")
                except Exception as e:
                    self.logger.error(f"Collection step {step_name} failed: {e}")
                    self.current_workflow.errors.append({
                        "phase": "collection",
                        "step": step_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        return all_resources
    
    def _execute_analysis(self, resources: List[BaseResource]) -> List[AnalysisResult]:
        """Execute analysis phase"""
        results = []
        
        for analyzer in self.analyzers:
            try:
                self.logger.info(f"Running analyzer: {analyzer.name}")
                result = analyzer.analyze(resources)
                results.append(result)
                self.logger.info(f"Analyzer {analyzer.name}: found {len(result.optimizations)} optimizations")
            except Exception as e:
                self.logger.error(f"Analyzer {analyzer.name} failed: {e}")
                self.current_workflow.errors.append({
                    "phase": "analysis",
                    "analyzer": analyzer.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def _execute_optimization_planning(self, analysis_results: List[AnalysisResult]) -> OptimizationPlan:
        """Execute optimization planning phase"""
        
        # Combine all optimizations
        all_optimizations = []
        for result in analysis_results:
            all_optimizations.extend(result.optimizations)
        
        # Sort by savings potential
        all_optimizations.sort(key=lambda x: x.savings, reverse=True)
        
        # Create optimization plan
        if self.optimizer:
            plan = self.optimizer.create_optimization_plan(
                all_optimizations,
                name=f"Optimization Plan - {datetime.now().strftime('%Y-%m-%d')}"
            )
            return plan
        else:
            # Create a basic plan without optimizer
            from ..base import OptimizationPlan
            return OptimizationPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Basic Optimization Plan",
                description="Optimization opportunities identified",
                optimizations=all_optimizations,
                actions=[],
                total_savings=sum(opt.savings for opt in all_optimizations),
                total_effort_hours=sum(opt.estimated_effort_hours for opt in all_optimizations),
                created_at=datetime.now()
            )
    
    def save_workflow_result(self, filepath: str):
        """Save workflow result to file"""
        if self.current_workflow:
            with open(filepath, 'w') as f:
                json.dump(self.current_workflow.to_dict(), f, indent=2)
            self.logger.info(f"Saved workflow result to {filepath}")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of current workflow"""
        if not self.current_workflow:
            return {}
        
        return {
            "workflow_id": self.current_workflow.workflow_id,
            "status": "completed" if self.current_workflow.success else "failed",
            "duration_seconds": (
                self.current_workflow.completed_at - self.current_workflow.started_at
            ).total_seconds() if self.current_workflow.completed_at else 0,
            "resources_collected": len(self.current_workflow.resources_collected),
            "optimizations_found": sum(
                len(r.optimizations) for r in self.current_workflow.analysis_results
            ),
            "total_monthly_savings": sum(
                r.total_monthly_savings for r in self.current_workflow.analysis_results
            ),
            "total_annual_savings": sum(
                r.total_annual_savings for r in self.current_workflow.analysis_results
            ),
            "errors": len(self.current_workflow.errors)
        }