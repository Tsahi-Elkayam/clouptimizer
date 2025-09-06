"""
Advanced Optimization Engine with ML Predictions
Uses machine learning and statistical analysis for intelligent cost optimization.
Provides predictive analytics, anomaly detection, and multi-resource dependency analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import json
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    AGGRESSIVE = "aggressive"  # Maximum savings, higher risk
    BALANCED = "balanced"      # Balance savings and stability
    CONSERVATIVE = "conservative"  # Minimal risk, moderate savings
    CUSTOM = "custom"          # User-defined parameters


class PredictionConfidence(str, Enum):
    """Prediction confidence levels"""
    VERY_HIGH = "very_high"  # >90% confidence
    HIGH = "high"            # 75-90% confidence
    MEDIUM = "medium"        # 60-75% confidence
    LOW = "low"             # <60% confidence


@dataclass
class CostPrediction:
    """Cost prediction result"""
    resource_id: str
    resource_type: str
    current_cost: float
    predicted_cost_30d: float
    predicted_cost_90d: float
    predicted_cost_1y: float
    confidence: PredictionConfidence
    trend: str  # increasing, stable, decreasing
    anomaly_score: float  # 0-1, higher means more anomalous
    factors: List[str]  # Factors influencing prediction
    recommendations: List[str]
    
    @property
    def cost_increase_30d(self) -> float:
        """Calculate cost increase in 30 days"""
        return self.predicted_cost_30d - self.current_cost
    
    @property
    def cost_increase_percentage(self) -> float:
        """Calculate percentage cost increase"""
        if self.current_cost > 0:
            return (self.cost_increase_30d / self.current_cost) * 100
        return 0


@dataclass
class ResourceCluster:
    """Cluster of related resources"""
    cluster_id: str
    resources: List[str]
    cluster_type: str  # application, environment, service
    total_cost: float
    optimization_potential: float
    dependencies: List[str]
    risk_score: float  # 0-1, higher means riskier to optimize
    
    @property
    def resource_count(self) -> int:
        return len(self.resources)


@dataclass
class OptimizationScenario:
    """Multi-resource optimization scenario"""
    scenario_id: str
    name: str
    description: str
    affected_resources: List[str]
    current_cost: float
    optimized_cost: float
    monthly_savings: float
    implementation_complexity: str  # low, medium, high
    risk_level: str  # low, medium, high
    dependencies: Dict[str, List[str]]
    implementation_order: List[str]
    rollback_plan: List[str]
    success_probability: float  # 0-1
    
    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12
    
    @property
    def savings_percentage(self) -> float:
        if self.current_cost > 0:
            return (self.monthly_savings / self.current_cost) * 100
        return 0


class AdvancedOptimizationEngine:
    """
    Advanced optimization engine using ML for intelligent cost optimization.
    Provides predictive analytics, pattern recognition, and scenario modeling.
    """
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        """
        Initialize Advanced Optimization Engine.
        
        Args:
            strategy: Optimization strategy to use
        """
        self.strategy = strategy
        self.cost_predictor = None
        self.anomaly_detector = None
        self.clustering_model = None
        self.resource_clusters: List[ResourceCluster] = []
        self.predictions: List[CostPrediction] = []
        self.scenarios: List[OptimizationScenario] = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # Cost prediction model
        self.cost_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Anomaly detection model
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        
        # Clustering model for resource grouping
        self.clustering_model = DBSCAN(
            eps=0.3,
            min_samples=2
        )
        
        # Time series model for trend analysis
        self.trend_model = LinearRegression()
        
        # Feature scaler
        self.scaler = StandardScaler()
    
    def train_models(self, historical_data: List[Dict]) -> None:
        """
        Train ML models with historical data.
        
        Args:
            historical_data: Historical cost and usage data
        """
        if len(historical_data) < 30:
            logger.warning("Insufficient data for training. Need at least 30 days of history.")
            return
        
        # Prepare training data
        X, y = self._prepare_training_data(historical_data)
        
        if X.shape[0] > 0:
            # Train cost predictor
            self.cost_predictor.fit(X, y)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X)
            
            logger.info(f"Models trained with {X.shape[0]} samples")
    
    def predict_costs(self, resources: List[Dict], days_ahead: int = 30) -> List[CostPrediction]:
        """
        Predict future costs for resources.
        
        Args:
            resources: List of resources to predict costs for
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of cost predictions
        """
        self.predictions = []
        
        for resource in resources:
            features = self._extract_features(resource)
            
            if features is not None and self.cost_predictor is not None:
                try:
                    # Scale features
                    features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
                    
                    # Predict costs
                    cost_30d = self.cost_predictor.predict(
                        self._adjust_features_for_time(features_scaled, 30)
                    )[0]
                    cost_90d = self.cost_predictor.predict(
                        self._adjust_features_for_time(features_scaled, 90)
                    )[0]
                    cost_1y = self.cost_predictor.predict(
                        self._adjust_features_for_time(features_scaled, 365)
                    )[0]
                    
                    # Detect anomalies
                    anomaly_score = self._calculate_anomaly_score(features_scaled)
                    
                    # Determine trend
                    trend = self._analyze_trend(resource)
                    
                    # Calculate confidence
                    confidence = self._calculate_prediction_confidence(
                        resource, features_scaled
                    )
                    
                    prediction = CostPrediction(
                        resource_id=resource.get('id', 'unknown'),
                        resource_type=resource.get('type', 'unknown'),
                        current_cost=resource.get('monthly_cost', 0),
                        predicted_cost_30d=cost_30d,
                        predicted_cost_90d=cost_90d,
                        predicted_cost_1y=cost_1y,
                        confidence=confidence,
                        trend=trend,
                        anomaly_score=anomaly_score,
                        factors=self._identify_cost_factors(resource),
                        recommendations=self._generate_predictions_recommendations(
                            resource, cost_30d, trend, anomaly_score
                        )
                    )
                    
                    self.predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Error predicting cost for {resource.get('id')}: {e}")
        
        return self.predictions
    
    def detect_anomalies(self, metrics: List[Dict]) -> List[Dict]:
        """
        Detect cost and usage anomalies.
        
        Args:
            metrics: Resource metrics data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not self.anomaly_detector:
            return anomalies
        
        for metric in metrics:
            features = self._extract_metric_features(metric)
            
            if features is not None:
                try:
                    features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
                    
                    # Predict anomaly
                    is_anomaly = self.anomaly_detector.predict(features_scaled)[0]
                    anomaly_score = self.anomaly_detector.score_samples(features_scaled)[0]
                    
                    if is_anomaly == -1:  # Anomaly detected
                        anomaly = {
                            'resource_id': metric.get('resource_id'),
                            'metric_name': metric.get('name'),
                            'value': metric.get('value'),
                            'expected_range': self._calculate_expected_range(metric),
                            'anomaly_score': abs(anomaly_score),
                            'severity': self._classify_anomaly_severity(anomaly_score),
                            'detected_at': datetime.now().isoformat(),
                            'recommendation': self._generate_anomaly_recommendation(metric)
                        }
                        anomalies.append(anomaly)
                        
                except Exception as e:
                    logger.error(f"Error detecting anomaly for {metric.get('resource_id')}: {e}")
        
        return anomalies
    
    def cluster_resources(self, resources: List[Dict]) -> List[ResourceCluster]:
        """
        Cluster related resources for group optimization.
        
        Args:
            resources: List of resources to cluster
            
        Returns:
            List of resource clusters
        """
        self.resource_clusters = []
        
        if len(resources) < 2:
            return self.resource_clusters
        
        # Extract features for clustering
        features = []
        resource_ids = []
        
        for resource in resources:
            feature = self._extract_clustering_features(resource)
            if feature is not None:
                features.append(feature)
                resource_ids.append(resource.get('id'))
        
        if len(features) < 2:
            return self.resource_clusters
        
        try:
            # Scale features
            features_array = np.array(features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Perform clustering
            clusters = self.clustering_model.fit_predict(features_scaled)
            
            # Group resources by cluster
            cluster_groups = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # -1 means noise/outlier in DBSCAN
                    cluster_groups[cluster_id].append(resource_ids[i])
            
            # Create ResourceCluster objects
            for cluster_id, resource_list in cluster_groups.items():
                cluster_resources = [r for r in resources 
                                   if r.get('id') in resource_list]
                
                cluster = ResourceCluster(
                    cluster_id=f"cluster-{cluster_id}",
                    resources=resource_list,
                    cluster_type=self._identify_cluster_type(cluster_resources),
                    total_cost=sum(r.get('monthly_cost', 0) for r in cluster_resources),
                    optimization_potential=self._calculate_cluster_optimization_potential(
                        cluster_resources
                    ),
                    dependencies=self._identify_dependencies(cluster_resources),
                    risk_score=self._calculate_cluster_risk(cluster_resources)
                )
                
                self.resource_clusters.append(cluster)
                
        except Exception as e:
            logger.error(f"Error clustering resources: {e}")
        
        return self.resource_clusters
    
    def generate_optimization_scenarios(self, 
                                       resources: List[Dict],
                                       constraints: Optional[Dict] = None) -> List[OptimizationScenario]:
        """
        Generate multi-resource optimization scenarios.
        
        Args:
            resources: List of resources to optimize
            constraints: Optimization constraints (budget, risk tolerance, etc.)
            
        Returns:
            List of optimization scenarios
        """
        self.scenarios = []
        
        # First, cluster resources
        clusters = self.cluster_resources(resources)
        
        # Generate scenarios based on strategy
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            self._generate_aggressive_scenarios(clusters, resources, constraints)
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            self._generate_conservative_scenarios(clusters, resources, constraints)
        else:  # BALANCED
            self._generate_balanced_scenarios(clusters, resources, constraints)
        
        # Add single-resource scenarios for high-impact optimizations
        self._add_high_impact_scenarios(resources, constraints)
        
        # Sort scenarios by savings potential
        self.scenarios.sort(key=lambda x: x.monthly_savings, reverse=True)
        
        return self.scenarios
    
    def optimize_with_dependencies(self, resources: List[Dict]) -> Dict[str, Any]:
        """
        Optimize resources considering dependencies.
        
        Args:
            resources: List of resources to optimize
            
        Returns:
            Optimization plan with dependency order
        """
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(resources)
        
        # Topological sort for optimization order
        optimization_order = self._topological_sort(dependency_graph)
        
        # Generate optimization plan
        optimization_plan = {
            'total_resources': len(resources),
            'optimization_order': optimization_order,
            'phases': [],
            'estimated_savings': 0,
            'estimated_duration_days': 0,
            'risk_assessment': {}
        }
        
        # Group into phases based on dependencies
        phases = self._group_into_phases(optimization_order, dependency_graph)
        
        for i, phase in enumerate(phases):
            phase_data = {
                'phase_number': i + 1,
                'resources': phase,
                'can_parallel': len(phase) > 1,
                'estimated_duration_days': self._estimate_phase_duration(phase),
                'estimated_savings': self._calculate_phase_savings(phase, resources),
                'prerequisites': self._get_phase_prerequisites(phase, phases[:i]),
                'rollback_plan': self._generate_rollback_plan(phase)
            }
            optimization_plan['phases'].append(phase_data)
            optimization_plan['estimated_savings'] += phase_data['estimated_savings']
            optimization_plan['estimated_duration_days'] += phase_data['estimated_duration_days']
        
        # Risk assessment
        optimization_plan['risk_assessment'] = self._assess_plan_risk(optimization_plan)
        
        return optimization_plan
    
    def calculate_optimization_impact(self, 
                                     scenario: OptimizationScenario,
                                     resources: List[Dict]) -> Dict[str, Any]:
        """
        Calculate the full impact of an optimization scenario.
        
        Args:
            scenario: Optimization scenario to evaluate
            resources: Current resources
            
        Returns:
            Impact analysis
        """
        impact = {
            'financial': {
                'current_monthly_cost': scenario.current_cost,
                'optimized_monthly_cost': scenario.optimized_cost,
                'monthly_savings': scenario.monthly_savings,
                'annual_savings': scenario.annual_savings,
                'roi_months': self._calculate_roi_period(scenario),
                'break_even_date': self._calculate_break_even_date(scenario)
            },
            'operational': {
                'affected_resources': len(scenario.affected_resources),
                'downtime_required': self._estimate_downtime(scenario),
                'complexity': scenario.implementation_complexity,
                'team_effort_hours': self._estimate_effort_hours(scenario)
            },
            'risk': {
                'risk_level': scenario.risk_level,
                'success_probability': scenario.success_probability,
                'rollback_complexity': self._assess_rollback_complexity(scenario),
                'potential_issues': self._identify_potential_issues(scenario)
            },
            'performance': {
                'expected_latency_change': self._estimate_latency_impact(scenario),
                'expected_throughput_change': self._estimate_throughput_impact(scenario),
                'expected_availability_change': self._estimate_availability_impact(scenario)
            },
            'compliance': {
                'data_residency_impact': self._assess_data_residency_impact(scenario),
                'security_impact': self._assess_security_impact(scenario),
                'audit_impact': self._assess_audit_impact(scenario)
            }
        }
        
        return impact
    
    # ML Helper Methods
    def _prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        features = []
        targets = []
        
        for data in historical_data:
            feature = self._extract_features(data)
            if feature is not None:
                features.append(feature)
                targets.append(data.get('cost', 0))
        
        if features:
            return np.array(features), np.array(targets)
        return np.array([]), np.array([])
    
    def _extract_features(self, resource: Dict) -> Optional[np.ndarray]:
        """Extract features from resource for ML"""
        try:
            features = [
                resource.get('cpu_utilization', 0),
                resource.get('memory_utilization', 0),
                resource.get('network_in', 0),
                resource.get('network_out', 0),
                resource.get('storage_used', 0),
                resource.get('request_count', 0),
                resource.get('age_days', 0),
                resource.get('instance_count', 1),
                self._encode_resource_type(resource.get('type', 'unknown')),
                self._encode_region(resource.get('region', 'us-east-1')),
                resource.get('monthly_cost', 0)
            ]
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _extract_metric_features(self, metric: Dict) -> Optional[np.ndarray]:
        """Extract features from metric for anomaly detection"""
        try:
            features = [
                metric.get('value', 0),
                metric.get('average', 0),
                metric.get('min', 0),
                metric.get('max', 0),
                metric.get('std_dev', 0),
                metric.get('count', 0),
                self._encode_metric_type(metric.get('name', 'unknown'))
            ]
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting metric features: {e}")
            return None
    
    def _extract_clustering_features(self, resource: Dict) -> Optional[np.ndarray]:
        """Extract features for resource clustering"""
        try:
            features = [
                self._encode_resource_type(resource.get('type', 'unknown')),
                self._encode_region(resource.get('region', 'us-east-1')),
                resource.get('monthly_cost', 0),
                resource.get('cpu_utilization', 0),
                resource.get('age_days', 0),
                len(resource.get('tags', {})),
                1 if resource.get('environment') == 'production' else 0
            ]
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting clustering features: {e}")
            return None
    
    def _adjust_features_for_time(self, features: np.ndarray, days: int) -> np.ndarray:
        """Adjust features for future time prediction"""
        adjusted = features.copy()
        # Add time component
        time_factor = days / 30  # Normalize to months
        adjusted = np.append(adjusted, [[time_factor]], axis=1)
        return adjusted
    
    def _calculate_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score"""
        try:
            if self.anomaly_detector:
                score = self.anomaly_detector.score_samples(features)[0]
                # Normalize to 0-1 range
                return 1 / (1 + np.exp(score))
            return 0
        except:
            return 0
    
    def _analyze_trend(self, resource: Dict) -> str:
        """Analyze cost trend"""
        history = resource.get('cost_history', [])
        if len(history) < 3:
            return "stable"
        
        # Fit linear regression to detect trend
        X = np.arange(len(history)).reshape(-1, 1)
        y = np.array(history)
        
        try:
            self.trend_model.fit(X, y)
            slope = self.trend_model.coef_[0]
            
            if slope > 0.05:
                return "increasing"
            elif slope < -0.05:
                return "decreasing"
            else:
                return "stable"
        except:
            return "stable"
    
    def _calculate_prediction_confidence(self, resource: Dict, features: np.ndarray) -> PredictionConfidence:
        """Calculate prediction confidence"""
        # Based on data quality and model performance
        data_points = len(resource.get('cost_history', []))
        variance = np.var(resource.get('cost_history', [0]))
        
        if data_points > 90 and variance < 0.1:
            return PredictionConfidence.VERY_HIGH
        elif data_points > 30 and variance < 0.2:
            return PredictionConfidence.HIGH
        elif data_points > 7 and variance < 0.3:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _identify_cost_factors(self, resource: Dict) -> List[str]:
        """Identify factors influencing cost"""
        factors = []
        
        if resource.get('cpu_utilization', 0) > 80:
            factors.append("High CPU utilization")
        if resource.get('storage_growth_rate', 0) > 10:
            factors.append("Rapid storage growth")
        if resource.get('network_out', 0) > 1000:
            factors.append("High data transfer")
        if resource.get('age_days', 0) > 365:
            factors.append("Aging infrastructure")
        
        return factors
    
    def _generate_predictions_recommendations(self, resource: Dict, predicted_cost: float,
                                             trend: str, anomaly_score: float) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        if trend == "increasing" and predicted_cost > resource.get('monthly_cost', 0) * 1.2:
            recommendations.append("Consider implementing cost controls")
            recommendations.append("Review and optimize resource utilization")
        
        if anomaly_score > 0.7:
            recommendations.append("Investigate unusual cost pattern")
            recommendations.append("Check for misconfigurations or waste")
        
        if predicted_cost > 1000:
            recommendations.append("Consider Reserved Instances or Savings Plans")
        
        return recommendations
    
    def _encode_resource_type(self, resource_type: str) -> float:
        """Encode resource type as numeric value"""
        types = {
            'ec2': 1, 'rds': 2, 's3': 3, 'lambda': 4,
            'dynamodb': 5, 'ecs': 6, 'eks': 7, 'unknown': 0
        }
        return types.get(resource_type.lower(), 0)
    
    def _encode_region(self, region: str) -> float:
        """Encode region as numeric value"""
        regions = {
            'us-east-1': 1, 'us-west-2': 2, 'eu-west-1': 3,
            'ap-southeast-1': 4, 'ap-northeast-1': 5, 'unknown': 0
        }
        return regions.get(region, 0)
    
    def _encode_metric_type(self, metric_type: str) -> float:
        """Encode metric type as numeric value"""
        types = {
            'cpu': 1, 'memory': 2, 'network': 3, 'disk': 4,
            'requests': 5, 'errors': 6, 'latency': 7, 'unknown': 0
        }
        return types.get(metric_type.lower(), 0)
    
    # Clustering and Scenario Methods
    def _identify_cluster_type(self, resources: List[Dict]) -> str:
        """Identify the type of resource cluster"""
        # Check common tags or patterns
        environments = set(r.get('environment', '') for r in resources)
        if len(environments) == 1:
            return "environment"
        
        applications = set(r.get('application', '') for r in resources)
        if len(applications) == 1:
            return "application"
        
        services = set(r.get('service', '') for r in resources)
        if len(services) == 1:
            return "service"
        
        return "mixed"
    
    def _calculate_cluster_optimization_potential(self, resources: List[Dict]) -> float:
        """Calculate optimization potential for a cluster"""
        total_cost = sum(r.get('monthly_cost', 0) for r in resources)
        total_waste = sum(r.get('estimated_waste', 0) for r in resources)
        
        if total_cost > 0:
            return (total_waste / total_cost) * 100
        return 0
    
    def _identify_dependencies(self, resources: List[Dict]) -> List[str]:
        """Identify dependencies between resources"""
        dependencies = []
        
        for resource in resources:
            deps = resource.get('dependencies', [])
            dependencies.extend(deps)
        
        return list(set(dependencies))
    
    def _calculate_cluster_risk(self, resources: List[Dict]) -> float:
        """Calculate risk score for optimizing a cluster"""
        if not resources:
            return 0
        
        # Factors that increase risk
        risk_factors = {
            'production': 0.3,
            'critical': 0.3,
            'customer_facing': 0.2,
            'data_processing': 0.1,
            'stateful': 0.1
        }
        
        risk_score = 0
        for resource in resources:
            for factor, weight in risk_factors.items():
                if resource.get(factor, False):
                    risk_score += weight
        
        # Normalize to 0-1
        return min(risk_score / len(resources), 1.0)
    
    def _generate_aggressive_scenarios(self, clusters: List[ResourceCluster],
                                      resources: List[Dict],
                                      constraints: Optional[Dict]) -> None:
        """Generate aggressive optimization scenarios"""
        for cluster in clusters:
            if cluster.optimization_potential > 20:  # >20% potential savings
                cluster_resources = [r for r in resources 
                                   if r.get('id') in cluster.resources]
                
                scenario = OptimizationScenario(
                    scenario_id=f"aggressive-{cluster.cluster_id}",
                    name=f"Aggressive optimization for {cluster.cluster_type}",
                    description="Maximum cost reduction with acceptable risk",
                    affected_resources=cluster.resources,
                    current_cost=cluster.total_cost,
                    optimized_cost=cluster.total_cost * 0.5,  # Target 50% reduction
                    monthly_savings=cluster.total_cost * 0.5,
                    implementation_complexity="high",
                    risk_level="medium",
                    dependencies=self._map_dependencies(cluster_resources),
                    implementation_order=self._determine_implementation_order(cluster.resources),
                    rollback_plan=self._create_rollback_plan(cluster.resources),
                    success_probability=0.75
                )
                
                self.scenarios.append(scenario)
    
    def _generate_conservative_scenarios(self, clusters: List[ResourceCluster],
                                        resources: List[Dict],
                                        constraints: Optional[Dict]) -> None:
        """Generate conservative optimization scenarios"""
        for cluster in clusters:
            if cluster.risk_score < 0.3:  # Low risk clusters only
                cluster_resources = [r for r in resources 
                                   if r.get('id') in cluster.resources]
                
                scenario = OptimizationScenario(
                    scenario_id=f"conservative-{cluster.cluster_id}",
                    name=f"Safe optimization for {cluster.cluster_type}",
                    description="Low-risk cost optimization",
                    affected_resources=cluster.resources[:5],  # Limit scope
                    current_cost=cluster.total_cost,
                    optimized_cost=cluster.total_cost * 0.8,  # Target 20% reduction
                    monthly_savings=cluster.total_cost * 0.2,
                    implementation_complexity="low",
                    risk_level="low",
                    dependencies=self._map_dependencies(cluster_resources[:5]),
                    implementation_order=self._determine_implementation_order(cluster.resources[:5]),
                    rollback_plan=self._create_rollback_plan(cluster.resources[:5]),
                    success_probability=0.95
                )
                
                self.scenarios.append(scenario)
    
    def _generate_balanced_scenarios(self, clusters: List[ResourceCluster],
                                    resources: List[Dict],
                                    constraints: Optional[Dict]) -> None:
        """Generate balanced optimization scenarios"""
        for cluster in clusters:
            if cluster.optimization_potential > 10 and cluster.risk_score < 0.5:
                cluster_resources = [r for r in resources 
                                   if r.get('id') in cluster.resources]
                
                scenario = OptimizationScenario(
                    scenario_id=f"balanced-{cluster.cluster_id}",
                    name=f"Balanced optimization for {cluster.cluster_type}",
                    description="Balanced approach to cost and risk",
                    affected_resources=cluster.resources,
                    current_cost=cluster.total_cost,
                    optimized_cost=cluster.total_cost * 0.65,  # Target 35% reduction
                    monthly_savings=cluster.total_cost * 0.35,
                    implementation_complexity="medium",
                    risk_level="medium",
                    dependencies=self._map_dependencies(cluster_resources),
                    implementation_order=self._determine_implementation_order(cluster.resources),
                    rollback_plan=self._create_rollback_plan(cluster.resources),
                    success_probability=0.85
                )
                
                self.scenarios.append(scenario)
    
    def _add_high_impact_scenarios(self, resources: List[Dict],
                                  constraints: Optional[Dict]) -> None:
        """Add scenarios for high-impact single resources"""
        for resource in resources:
            if resource.get('monthly_cost', 0) > 1000:  # High-cost resources
                scenario = OptimizationScenario(
                    scenario_id=f"high-impact-{resource.get('id')}",
                    name=f"Optimize high-cost {resource.get('type')}",
                    description=f"Targeted optimization for expensive resource",
                    affected_resources=[resource.get('id')],
                    current_cost=resource.get('monthly_cost', 0),
                    optimized_cost=resource.get('monthly_cost', 0) * 0.6,
                    monthly_savings=resource.get('monthly_cost', 0) * 0.4,
                    implementation_complexity="low",
                    risk_level="low",
                    dependencies={},
                    implementation_order=[resource.get('id')],
                    rollback_plan=[f"Revert {resource.get('id')} to original configuration"],
                    success_probability=0.9
                )
                
                self.scenarios.append(scenario)
    
    # Dependency Management Methods
    def _build_dependency_graph(self, resources: List[Dict]) -> Dict[str, Set[str]]:
        """Build dependency graph for resources"""
        graph = defaultdict(set)
        
        for resource in resources:
            resource_id = resource.get('id')
            dependencies = resource.get('dependencies', [])
            
            for dep in dependencies:
                graph[resource_id].add(dep)
        
        return dict(graph)
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Perform topological sort on dependency graph"""
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        queue = deque([node for node in graph if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _group_into_phases(self, order: List[str], graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Group resources into optimization phases"""
        phases = []
        processed = set()
        
        for resource in order:
            if resource in processed:
                continue
            
            # Find all resources that can be processed in parallel
            phase = [resource]
            processed.add(resource)
            
            for other in order:
                if other not in processed:
                    # Check if they have dependencies on each other
                    if (other not in graph.get(resource, set()) and
                        resource not in graph.get(other, set())):
                        phase.append(other)
                        processed.add(other)
            
            phases.append(phase)
        
        return phases
    
    # Impact Calculation Methods
    def _calculate_roi_period(self, scenario: OptimizationScenario) -> int:
        """Calculate ROI period in months"""
        if scenario.monthly_savings <= 0:
            return 999
        
        # Estimate implementation cost
        impl_cost = self._estimate_implementation_cost(scenario)
        
        return int(impl_cost / scenario.monthly_savings) + 1
    
    def _calculate_break_even_date(self, scenario: OptimizationScenario) -> str:
        """Calculate break-even date"""
        roi_months = self._calculate_roi_period(scenario)
        break_even = datetime.now() + timedelta(days=roi_months * 30)
        return break_even.strftime("%Y-%m-%d")
    
    def _estimate_downtime(self, scenario: OptimizationScenario) -> str:
        """Estimate required downtime"""
        if scenario.implementation_complexity == "low":
            return "0-30 minutes"
        elif scenario.implementation_complexity == "medium":
            return "1-4 hours"
        else:
            return "4-24 hours"
    
    def _estimate_effort_hours(self, scenario: OptimizationScenario) -> int:
        """Estimate team effort in hours"""
        base_hours = {
            "low": 4,
            "medium": 16,
            "high": 40
        }
        
        hours = base_hours.get(scenario.implementation_complexity, 20)
        hours += len(scenario.affected_resources) * 2  # Additional time per resource
        
        return hours
    
    def _estimate_implementation_cost(self, scenario: OptimizationScenario) -> float:
        """Estimate implementation cost"""
        effort_hours = self._estimate_effort_hours(scenario)
        hourly_rate = 150  # Assumed hourly rate
        
        return effort_hours * hourly_rate
    
    # Helper Methods
    def _map_dependencies(self, resources: List[Dict]) -> Dict[str, List[str]]:
        """Map dependencies for resources"""
        dep_map = {}
        for resource in resources:
            dep_map[resource.get('id')] = resource.get('dependencies', [])
        return dep_map
    
    def _determine_implementation_order(self, resource_ids: List[str]) -> List[str]:
        """Determine optimal implementation order"""
        # Simple ordering - in production, use dependency analysis
        return resource_ids
    
    def _create_rollback_plan(self, resource_ids: List[str]) -> List[str]:
        """Create rollback plan for resources"""
        plan = []
        for resource_id in resource_ids:
            plan.append(f"Restore {resource_id} from backup")
            plan.append(f"Verify {resource_id} functionality")
        plan.append("Validate all service connections")
        return plan
    
    def _calculate_expected_range(self, metric: Dict) -> Tuple[float, float]:
        """Calculate expected range for metric"""
        avg = metric.get('average', 0)
        std = metric.get('std_dev', 1)
        
        return (avg - 2 * std, avg + 2 * std)
    
    def _classify_anomaly_severity(self, score: float) -> str:
        """Classify anomaly severity"""
        if abs(score) > 0.8:
            return "critical"
        elif abs(score) > 0.6:
            return "high"
        elif abs(score) > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_anomaly_recommendation(self, metric: Dict) -> str:
        """Generate recommendation for anomaly"""
        if metric.get('name') == 'cpu' and metric.get('value', 0) > 90:
            return "Consider scaling up or optimizing application"
        elif metric.get('name') == 'cost' and metric.get('value', 0) > metric.get('average', 0) * 2:
            return "Investigate unusual cost spike"
        else:
            return "Review resource configuration and usage patterns"
    
    def _estimate_phase_duration(self, phase: List[str]) -> int:
        """Estimate phase duration in days"""
        return max(1, len(phase) // 5)  # 5 resources per day
    
    def _calculate_phase_savings(self, phase: List[str], resources: List[Dict]) -> float:
        """Calculate savings for a phase"""
        phase_resources = [r for r in resources if r.get('id') in phase]
        return sum(r.get('estimated_savings', 0) for r in phase_resources)
    
    def _get_phase_prerequisites(self, phase: List[str], previous_phases: List[List[str]]) -> List[str]:
        """Get prerequisites for a phase"""
        prerequisites = []
        for prev_phase in previous_phases:
            prerequisites.extend(prev_phase)
        return prerequisites
    
    def _assess_plan_risk(self, plan: Dict) -> Dict[str, Any]:
        """Assess risk for optimization plan"""
        total_resources = plan['total_resources']
        total_phases = len(plan['phases'])
        
        risk_score = min(1.0, (total_resources * 0.05 + total_phases * 0.1))
        
        return {
            'overall_risk': "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            'risk_score': risk_score,
            'risk_factors': self._identify_risk_factors(plan),
            'mitigation_strategies': self._suggest_mitigation_strategies(risk_score)
        }
    
    def _identify_risk_factors(self, plan: Dict) -> List[str]:
        """Identify risk factors in plan"""
        factors = []
        
        if plan['total_resources'] > 50:
            factors.append("Large number of resources")
        if plan['estimated_duration_days'] > 30:
            factors.append("Long implementation timeline")
        if any(p.get('can_parallel') for p in plan['phases']):
            factors.append("Parallel implementation complexity")
        
        return factors
    
    def _suggest_mitigation_strategies(self, risk_score: float) -> List[str]:
        """Suggest risk mitigation strategies"""
        strategies = []
        
        if risk_score > 0.7:
            strategies.append("Implement in smaller batches")
            strategies.append("Increase testing coverage")
            strategies.append("Have rollback plan ready")
        elif risk_score > 0.4:
            strategies.append("Perform thorough testing")
            strategies.append("Monitor closely during implementation")
        else:
            strategies.append("Standard monitoring and validation")
        
        return strategies
    
    def _assess_rollback_complexity(self, scenario: OptimizationScenario) -> str:
        """Assess rollback complexity"""
        if len(scenario.affected_resources) > 20:
            return "high"
        elif len(scenario.affected_resources) > 5:
            return "medium"
        else:
            return "low"
    
    def _identify_potential_issues(self, scenario: OptimizationScenario) -> List[str]:
        """Identify potential issues"""
        issues = []
        
        if scenario.implementation_complexity == "high":
            issues.append("Complex implementation may have unexpected issues")
        if len(scenario.dependencies) > 10:
            issues.append("Many dependencies increase failure risk")
        if scenario.risk_level == "high":
            issues.append("High risk of service disruption")
        
        return issues
    
    def _estimate_latency_impact(self, scenario: OptimizationScenario) -> str:
        """Estimate latency impact"""
        if "region" in str(scenario.description).lower():
            return "+10-50ms for cross-region"
        elif "instance" in str(scenario.description).lower():
            return "+5-10ms during transition"
        else:
            return "Minimal impact expected"
    
    def _estimate_throughput_impact(self, scenario: OptimizationScenario) -> str:
        """Estimate throughput impact"""
        if scenario.implementation_complexity == "high":
            return "-20% during implementation"
        else:
            return "No significant impact"
    
    def _estimate_availability_impact(self, scenario: OptimizationScenario) -> str:
        """Estimate availability impact"""
        downtime = self._estimate_downtime(scenario)
        if "0-30 minutes" in downtime:
            return "99.9% maintained"
        else:
            return "99.5% during implementation"
    
    def _assess_data_residency_impact(self, scenario: OptimizationScenario) -> str:
        """Assess data residency impact"""
        if "region" in str(scenario.description).lower():
            return "Verify compliance with data residency requirements"
        return "No impact"
    
    def _assess_security_impact(self, scenario: OptimizationScenario) -> str:
        """Assess security impact"""
        if scenario.risk_level == "high":
            return "Review security configurations"
        return "Standard security maintained"
    
    def _assess_audit_impact(self, scenario: OptimizationScenario) -> str:
        """Assess audit impact"""
        return "Document all changes for audit trail"