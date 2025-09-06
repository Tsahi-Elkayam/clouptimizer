"""
Quick Wins Analyzer - Immediate High-Impact Optimizations
Identifies "low-hanging fruit" for immediate cost savings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ImplementationEffort(str, Enum):
    """Implementation effort levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(str, Enum):
    """Risk levels for optimizations"""
    ZERO = "ZERO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Confidence(str, Enum):
    """Confidence levels for recommendations"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class QuickWin:
    """Represents a quick optimization opportunity"""
    resource_id: str
    resource_type: str
    region: str
    category: str
    title: str
    description: str
    current_monthly_cost: float
    potential_savings: float
    savings_percentage: float
    implementation_effort: ImplementationEffort
    risk_level: RiskLevel
    action_steps: List[str]
    confidence: Confidence
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def annual_savings(self) -> float:
        """Calculate annual savings"""
        return self.potential_savings * 12
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score based on savings, effort, and risk"""
        # Higher savings = higher score
        savings_score = min(self.potential_savings / 100, 10)  # Normalize to 0-10
        
        # Lower effort = higher score
        effort_multiplier = {
            ImplementationEffort.LOW: 3.0,
            ImplementationEffort.MEDIUM: 2.0,
            ImplementationEffort.HIGH: 1.0
        }[self.implementation_effort]
        
        # Lower risk = higher score
        risk_multiplier = {
            RiskLevel.ZERO: 2.0,
            RiskLevel.LOW: 1.5,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.5
        }[self.risk_level]
        
        return savings_score * effort_multiplier * risk_multiplier


class QuickWinsAnalyzer:
    """
    Identifies immediate, high-impact optimization opportunities.
    Focuses on easy wins that can be implemented quickly.
    """
    
    # Pricing constants for quick calculations (USD)
    PRICING = {
        'elastic_ip_unused': 3.60,  # Per month for unattached EIP
        'ebs_snapshot_gb': 0.05,  # Per GB per month
        'gp2_to_gp3_savings_rate': 0.20,  # 20% savings typically
        'cloudwatch_logs_gb': 0.50,  # Per GB ingested
        'cloudwatch_logs_storage_gb': 0.03,  # Per GB per month stored
        'old_ami_gb': 0.05,  # Per GB per month (snapshot backing)
        'nat_gateway_hourly': 0.045,  # Per hour
        'nat_gateway_monthly': 32.40,  # Per month
        'alb_hourly': 0.0225,  # Per hour
        'alb_monthly': 16.20,  # Per month
        'nlb_hourly': 0.0225,  # Per hour
        'vpc_endpoint_hourly': 0.01,  # Per hour
        'lambda_request_million': 0.20,  # Per million requests
        'lambda_gb_second': 0.0000166667,  # Per GB-second
    }
    
    # Quick win thresholds
    THRESHOLDS = {
        'old_snapshot_days': 90,
        'old_ami_days': 180,
        'log_retention_excessive_days': 90,
        'small_volume_gb': 100,
        'orphaned_volume_days': 7,
        'idle_resource_days': 14,
        'low_utilization_cpu': 10.0,  # CPU %
        'high_utilization_cpu': 80.0,  # CPU %
        'unused_elastic_ip_days': 1,
        'old_lambda_days': 90,
        'minimum_savings_threshold': 10.0,  # Minimum $10/month savings
        'unattached_volume_days': 7,
        'stopped_instance_days': 30,
        'old_backup_days': 90,
        'low_storage_utilization': 20.0,  # Storage %
    }
    
    def __init__(self, min_savings_threshold: float = 10.0):
        """
        Initialize Quick Wins Analyzer.
        
        Args:
            min_savings_threshold: Minimum monthly savings to consider (USD)
        """
        self.min_savings_threshold = min_savings_threshold
        self.quick_wins: List[QuickWin] = []
    
    def analyze_resources(self, resources: List[Dict[str, Any]]) -> List[QuickWin]:
        """
        Analyze resources for quick win opportunities.
        
        Args:
            resources: List of cloud resources to analyze
            
        Returns:
            List of quick win opportunities sorted by priority
        """
        self.quick_wins = []
        
        # Group resources by type for efficient analysis
        resources_by_type = defaultdict(list)
        for resource in resources:
            resource_type = resource.get('type', 'unknown')
            resources_by_type[resource_type].append(resource)
        
        # Analyze each resource type
        self._analyze_elastic_ips(resources_by_type.get('elastic_ip', []))
        self._analyze_ebs_volumes(resources_by_type.get('ebs_volume', []))
        self._analyze_ebs_snapshots(resources_by_type.get('ebs_snapshot', []))
        self._analyze_ec2_instances(resources_by_type.get('ec2_instance', []))
        self._analyze_rds_instances(resources_by_type.get('rds_instance', []))
        self._analyze_load_balancers(resources_by_type.get('load_balancer', []))
        self._analyze_nat_gateways(resources_by_type.get('nat_gateway', []))
        self._analyze_lambda_functions(resources_by_type.get('lambda_function', []))
        self._analyze_cloudwatch_logs(resources_by_type.get('log_group', []))
        self._analyze_s3_buckets(resources_by_type.get('s3_bucket', []))
        
        # Filter by minimum savings threshold
        self.quick_wins = [
            qw for qw in self.quick_wins 
            if qw.potential_savings >= self.min_savings_threshold
        ]
        
        # Sort by priority score
        self.quick_wins.sort(key=lambda x: x.priority_score, reverse=True)
        
        return self.quick_wins
    
    def _analyze_elastic_ips(self, resources: List[Dict]) -> None:
        """Analyze Elastic IPs for optimization"""
        for eip in resources:
            if not eip.get('association_id'):  # Unattached EIP
                days_unattached = self._get_resource_age_days(eip)
                if days_unattached >= self.THRESHOLDS['unused_elastic_ip_days']:
                    monthly_cost = self.PRICING['elastic_ip_unused']
                    
                    self.quick_wins.append(QuickWin(
                        resource_id=eip.get('allocation_id', 'unknown'),
                        resource_type='ElasticIP',
                        region=eip.get('region', 'unknown'),
                        category='Unused Resources',
                        title='Release Unattached Elastic IP',
                        description=f"Elastic IP has been unattached for {days_unattached} days",
                        current_monthly_cost=monthly_cost,
                        potential_savings=monthly_cost,
                        savings_percentage=100.0,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.ZERO,
                        action_steps=[
                            f"Release Elastic IP {eip.get('public_ip', 'unknown')}",
                            "Verify no automation depends on this IP",
                            "Document IP release for future reference"
                        ],
                        confidence=Confidence.HIGH,
                        tags=eip.get('tags', {}),
                        metadata={'public_ip': eip.get('public_ip'), 'days_unattached': days_unattached}
                    ))
    
    def _analyze_ebs_volumes(self, resources: List[Dict]) -> None:
        """Analyze EBS volumes for optimization"""
        for volume in resources:
            volume_size = volume.get('size', 0)
            volume_type = volume.get('volume_type', 'gp2')
            monthly_cost = self._calculate_ebs_cost(volume_size, volume_type)
            
            # Check for unattached volumes
            if volume.get('state') == 'available':
                days_unattached = self._get_resource_age_days(volume)
                if days_unattached >= self.THRESHOLDS['unattached_volume_days']:
                    self.quick_wins.append(QuickWin(
                        resource_id=volume.get('volume_id', 'unknown'),
                        resource_type='EBSVolume',
                        region=volume.get('availability_zone', 'unknown')[:-1],
                        category='Unused Resources',
                        title='Delete Unattached EBS Volume',
                        description=f"Volume ({volume_size}GB) unattached for {days_unattached} days",
                        current_monthly_cost=monthly_cost,
                        potential_savings=monthly_cost,
                        savings_percentage=100.0,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.LOW,
                        action_steps=[
                            f"Create snapshot of volume {volume.get('volume_id')} for backup",
                            "Verify volume is not needed",
                            "Delete the volume"
                        ],
                        confidence=Confidence.HIGH,
                        tags=volume.get('tags', {}),
                        metadata={'size_gb': volume_size, 'volume_type': volume_type}
                    ))
            
            # Check for gp2 to gp3 migration opportunity
            elif volume_type == 'gp2' and volume_size >= 100:
                savings = monthly_cost * self.PRICING['gp2_to_gp3_savings_rate']
                if savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=volume.get('volume_id', 'unknown'),
                        resource_type='EBSVolume',
                        region=volume.get('availability_zone', 'unknown')[:-1],
                        category='Storage Optimization',
                        title='Migrate GP2 to GP3 Volume',
                        description=f"Convert {volume_size}GB GP2 volume to GP3 for cost savings",
                        current_monthly_cost=monthly_cost,
                        potential_savings=savings,
                        savings_percentage=self.PRICING['gp2_to_gp3_savings_rate'] * 100,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.ZERO,
                        action_steps=[
                            "Modify volume type from gp2 to gp3",
                            "No downtime required",
                            "Performance will be same or better"
                        ],
                        confidence=Confidence.HIGH,
                        tags=volume.get('tags', {}),
                        metadata={'size_gb': volume_size, 'current_type': 'gp2', 'target_type': 'gp3'}
                    ))
    
    def _analyze_ebs_snapshots(self, resources: List[Dict]) -> None:
        """Analyze EBS snapshots for optimization"""
        # Group snapshots by volume
        snapshots_by_volume = defaultdict(list)
        for snapshot in resources:
            volume_id = snapshot.get('volume_id', 'orphaned')
            snapshots_by_volume[volume_id].append(snapshot)
        
        for volume_id, snapshots in snapshots_by_volume.items():
            # Sort by creation date
            snapshots.sort(key=lambda x: x.get('start_time', ''), reverse=True)
            
            # Keep recent snapshots, flag old ones
            old_snapshots = []
            total_size_gb = 0
            
            for snapshot in snapshots[3:]:  # Keep latest 3 snapshots
                age_days = self._get_resource_age_days(snapshot)
                if age_days > self.THRESHOLDS['old_snapshot_days']:
                    old_snapshots.append(snapshot)
                    total_size_gb += snapshot.get('volume_size', 0)
            
            if old_snapshots and total_size_gb > 0:
                monthly_cost = total_size_gb * self.PRICING['ebs_snapshot_gb']
                if monthly_cost >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=f"volume-{volume_id}-snapshots",
                        resource_type='EBSSnapshot',
                        region=old_snapshots[0].get('region', 'unknown'),
                        category='Backup Optimization',
                        title=f'Delete {len(old_snapshots)} Old Snapshots',
                        description=f"Delete snapshots older than {self.THRESHOLDS['old_snapshot_days']} days ({total_size_gb}GB total)",
                        current_monthly_cost=monthly_cost,
                        potential_savings=monthly_cost,
                        savings_percentage=100.0,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.LOW,
                        action_steps=[
                            f"Review {len(old_snapshots)} old snapshots",
                            "Verify snapshots are not needed for compliance",
                            "Delete snapshots older than retention policy"
                        ],
                        confidence=Confidence.HIGH,
                        metadata={
                            'snapshot_count': len(old_snapshots),
                            'total_size_gb': total_size_gb,
                            'volume_id': volume_id
                        }
                    ))
    
    def _analyze_ec2_instances(self, resources: List[Dict]) -> None:
        """Analyze EC2 instances for optimization"""
        for instance in resources:
            instance_id = instance.get('instance_id', 'unknown')
            instance_type = instance.get('instance_type', 'unknown')
            state = instance.get('state', {}).get('name', 'unknown')
            
            # Estimate monthly cost (simplified)
            monthly_cost = self._estimate_instance_cost(instance_type)
            
            # Check for stopped instances
            if state == 'stopped':
                days_stopped = self._get_state_duration_days(instance, 'stopped')
                if days_stopped >= self.THRESHOLDS['stopped_instance_days']:
                    self.quick_wins.append(QuickWin(
                        resource_id=instance_id,
                        resource_type='EC2Instance',
                        region=instance.get('region', 'unknown'),
                        category='Unused Resources',
                        title='Terminate Long-Stopped Instance',
                        description=f"Instance {instance_type} stopped for {days_stopped} days",
                        current_monthly_cost=monthly_cost * 0.1,  # Storage cost only
                        potential_savings=monthly_cost * 0.1,
                        savings_percentage=100.0,
                        implementation_effort=ImplementationEffort.MEDIUM,
                        risk_level=RiskLevel.MEDIUM,
                        action_steps=[
                            "Create AMI backup of instance",
                            "Verify instance is not needed",
                            "Terminate instance",
                            "Delete associated EBS volumes if not needed"
                        ],
                        confidence=Confidence.MEDIUM,
                        tags=instance.get('tags', {}),
                        metadata={'instance_type': instance_type, 'days_stopped': days_stopped}
                    ))
            
            # Check for low CPU utilization (running instances)
            elif state == 'running':
                cpu_utilization = instance.get('metrics', {}).get('cpu_utilization_avg', 100)
                if cpu_utilization < self.THRESHOLDS['low_utilization_cpu']:
                    # Suggest downsizing
                    smaller_type = self._get_smaller_instance_type(instance_type)
                    if smaller_type:
                        new_cost = self._estimate_instance_cost(smaller_type)
                        savings = monthly_cost - new_cost
                        if savings >= self.min_savings_threshold:
                            self.quick_wins.append(QuickWin(
                                resource_id=instance_id,
                                resource_type='EC2Instance',
                                region=instance.get('region', 'unknown'),
                                category='Rightsizing',
                                title='Downsize Underutilized Instance',
                                description=f"CPU utilization only {cpu_utilization:.1f}% - downsize from {instance_type} to {smaller_type}",
                                current_monthly_cost=monthly_cost,
                                potential_savings=savings,
                                savings_percentage=(savings / monthly_cost) * 100,
                                implementation_effort=ImplementationEffort.MEDIUM,
                                risk_level=RiskLevel.LOW,
                                action_steps=[
                                    f"Schedule maintenance window",
                                    f"Stop instance",
                                    f"Change instance type to {smaller_type}",
                                    f"Start instance and verify performance"
                                ],
                                confidence=Confidence.MEDIUM,
                                tags=instance.get('tags', {}),
                                metadata={
                                    'current_type': instance_type,
                                    'recommended_type': smaller_type,
                                    'cpu_utilization': cpu_utilization
                                }
                            ))
    
    def _analyze_rds_instances(self, resources: List[Dict]) -> None:
        """Analyze RDS instances for optimization"""
        for rds in resources:
            instance_class = rds.get('instance_class', 'unknown')
            engine = rds.get('engine', 'unknown')
            multi_az = rds.get('multi_az', False)
            storage_gb = rds.get('allocated_storage', 0)
            
            monthly_cost = self._estimate_rds_cost(instance_class, storage_gb, multi_az)
            
            # Check for idle RDS instances
            connections = rds.get('metrics', {}).get('database_connections_avg', 0)
            cpu_utilization = rds.get('metrics', {}).get('cpu_utilization_avg', 100)
            
            if connections < 1 and cpu_utilization < self.THRESHOLDS['low_utilization_cpu']:
                self.quick_wins.append(QuickWin(
                    resource_id=rds.get('db_instance_identifier', 'unknown'),
                    resource_type='RDSInstance',
                    region=rds.get('region', 'unknown'),
                    category='Unused Resources',
                    title='Delete or Stop Idle RDS Instance',
                    description=f"RDS instance with no connections and {cpu_utilization:.1f}% CPU",
                    current_monthly_cost=monthly_cost,
                    potential_savings=monthly_cost * 0.9,  # Can stop RDS to save 90%
                    savings_percentage=90.0,
                    implementation_effort=ImplementationEffort.MEDIUM,
                    risk_level=RiskLevel.MEDIUM,
                    action_steps=[
                        "Create final snapshot",
                        "Stop RDS instance (if supported by engine)",
                        "Or delete if not needed with final snapshot"
                    ],
                    confidence=Confidence.HIGH,
                    tags=rds.get('tags', {}),
                    metadata={
                        'instance_class': instance_class,
                        'engine': engine,
                        'connections': connections,
                        'cpu_utilization': cpu_utilization
                    }
                ))
            
            # Check for Multi-AZ optimization
            elif multi_az and cpu_utilization < 20:
                savings = monthly_cost * 0.4  # Multi-AZ roughly doubles cost
                if savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=rds.get('db_instance_identifier', 'unknown'),
                        resource_type='RDSInstance',
                        region=rds.get('region', 'unknown'),
                        category='High Availability',
                        title='Convert Multi-AZ to Single-AZ',
                        description=f"Low utilization RDS may not need Multi-AZ",
                        current_monthly_cost=monthly_cost,
                        potential_savings=savings,
                        savings_percentage=40.0,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.MEDIUM,
                        action_steps=[
                            "Evaluate high availability requirements",
                            "Schedule maintenance window",
                            "Modify RDS to disable Multi-AZ",
                            "Implement alternative backup strategy"
                        ],
                        confidence=Confidence.MEDIUM,
                        tags=rds.get('tags', {}),
                        metadata={'instance_class': instance_class, 'multi_az': True}
                    ))
    
    def _analyze_nat_gateways(self, resources: List[Dict]) -> None:
        """Analyze NAT Gateways for optimization"""
        # Group NAT gateways by VPC
        nat_by_vpc = defaultdict(list)
        for nat in resources:
            vpc_id = nat.get('vpc_id', 'unknown')
            nat_by_vpc[vpc_id].append(nat)
        
        for vpc_id, nats in nat_by_vpc.items():
            if len(nats) > 1:
                # Multiple NAT gateways in same VPC - potential consolidation
                monthly_cost_per_nat = self.PRICING['nat_gateway_monthly']
                potential_savings = monthly_cost_per_nat * (len(nats) - 1)
                
                if potential_savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=f"vpc-{vpc_id}-nats",
                        resource_type='NATGateway',
                        region=nats[0].get('region', 'unknown'),
                        category='Network Optimization',
                        title='Consolidate NAT Gateways',
                        description=f"VPC has {len(nats)} NAT gateways - consider consolidation",
                        current_monthly_cost=monthly_cost_per_nat * len(nats),
                        potential_savings=potential_savings,
                        savings_percentage=(potential_savings / (monthly_cost_per_nat * len(nats))) * 100,
                        implementation_effort=ImplementationEffort.HIGH,
                        risk_level=RiskLevel.MEDIUM,
                        action_steps=[
                            "Review high availability requirements",
                            "Analyze traffic patterns",
                            "Consider NAT instance for dev/test",
                            "Or use single NAT gateway with acceptance of AZ failure risk"
                        ],
                        confidence=Confidence.MEDIUM,
                        metadata={'vpc_id': vpc_id, 'nat_count': len(nats)}
                    ))
    
    def _analyze_load_balancers(self, resources: List[Dict]) -> None:
        """Analyze Load Balancers for optimization"""
        for lb in resources:
            lb_type = lb.get('type', 'application')
            lb_name = lb.get('name', 'unknown')
            
            # Check for idle load balancers
            target_count = lb.get('metrics', {}).get('healthy_targets', 0)
            request_count = lb.get('metrics', {}).get('request_count_daily', 0)
            
            monthly_cost = self.PRICING['alb_monthly'] if lb_type == 'application' else self.PRICING['alb_monthly']
            
            if target_count == 0 or request_count < 100:
                self.quick_wins.append(QuickWin(
                    resource_id=lb.get('arn', 'unknown'),
                    resource_type='LoadBalancer',
                    region=lb.get('region', 'unknown'),
                    category='Unused Resources',
                    title='Delete Idle Load Balancer',
                    description=f"{lb_type.upper()} with {target_count} targets and {request_count} daily requests",
                    current_monthly_cost=monthly_cost,
                    potential_savings=monthly_cost,
                    savings_percentage=100.0,
                    implementation_effort=ImplementationEffort.LOW,
                    risk_level=RiskLevel.LOW,
                    action_steps=[
                        "Verify load balancer is not needed",
                        "Update DNS if necessary",
                        "Delete load balancer"
                    ],
                    confidence=Confidence.HIGH,
                    tags=lb.get('tags', {}),
                    metadata={'lb_type': lb_type, 'name': lb_name}
                ))
    
    def _analyze_lambda_functions(self, resources: List[Dict]) -> None:
        """Analyze Lambda functions for optimization"""
        for lambda_func in resources:
            function_name = lambda_func.get('function_name', 'unknown')
            memory_size = lambda_func.get('memory_size', 128)
            
            # Check for overprovisioned memory
            max_memory_used = lambda_func.get('metrics', {}).get('max_memory_used_mb', 0)
            if max_memory_used > 0 and max_memory_used < memory_size * 0.5:
                # Function uses less than 50% of allocated memory
                recommended_memory = max(128, int(max_memory_used * 1.5))  # 50% buffer
                
                # Estimate savings (simplified)
                current_gb_seconds = lambda_func.get('metrics', {}).get('monthly_gb_seconds', 0)
                new_gb_seconds = current_gb_seconds * (recommended_memory / memory_size)
                monthly_savings = (current_gb_seconds - new_gb_seconds) * self.PRICING['lambda_gb_second']
                
                if monthly_savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=lambda_func.get('function_arn', 'unknown'),
                        resource_type='LambdaFunction',
                        region=lambda_func.get('region', 'unknown'),
                        category='Rightsizing',
                        title='Reduce Lambda Memory Allocation',
                        description=f"Function uses only {max_memory_used}MB of {memory_size}MB",
                        current_monthly_cost=current_gb_seconds * self.PRICING['lambda_gb_second'],
                        potential_savings=monthly_savings,
                        savings_percentage=(monthly_savings / (current_gb_seconds * self.PRICING['lambda_gb_second'])) * 100,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.LOW,
                        action_steps=[
                            f"Reduce memory from {memory_size}MB to {recommended_memory}MB",
                            "Test function performance",
                            "Monitor for cold start impact"
                        ],
                        confidence=Confidence.HIGH,
                        tags=lambda_func.get('tags', {}),
                        metadata={
                            'function_name': function_name,
                            'current_memory': memory_size,
                            'recommended_memory': recommended_memory
                        }
                    ))
    
    def _analyze_cloudwatch_logs(self, resources: List[Dict]) -> None:
        """Analyze CloudWatch Logs for optimization"""
        for log_group in resources:
            retention_days = log_group.get('retention_in_days', 0)
            stored_bytes = log_group.get('stored_bytes', 0)
            stored_gb = stored_bytes / (1024 ** 3)
            
            # Check for excessive retention
            if retention_days == 0 or retention_days > self.THRESHOLDS['log_retention_excessive_days']:
                monthly_cost = stored_gb * self.PRICING['cloudwatch_logs_storage_gb']
                
                if retention_days == 0:
                    # Never expire - highest cost
                    recommended_retention = 90
                    savings_percentage = 70  # Estimate 70% reduction with 90-day retention
                else:
                    recommended_retention = self.THRESHOLDS['log_retention_excessive_days']
                    savings_percentage = ((retention_days - recommended_retention) / retention_days) * 100
                
                potential_savings = monthly_cost * (savings_percentage / 100)
                
                if potential_savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=log_group.get('log_group_name', 'unknown'),
                        resource_type='CloudWatchLogs',
                        region=log_group.get('region', 'unknown'),
                        category='Storage Optimization',
                        title='Reduce Log Retention Period',
                        description=f"Reduce retention from {retention_days if retention_days > 0 else 'Never Expire'} to {recommended_retention} days",
                        current_monthly_cost=monthly_cost,
                        potential_savings=potential_savings,
                        savings_percentage=savings_percentage,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.LOW,
                        action_steps=[
                            f"Set retention to {recommended_retention} days",
                            "Archive important logs to S3 if needed",
                            "Update compliance documentation"
                        ],
                        confidence=Confidence.HIGH,
                        metadata={
                            'current_retention': retention_days,
                            'recommended_retention': recommended_retention,
                            'stored_gb': stored_gb
                        }
                    ))
    
    def _analyze_s3_buckets(self, resources: List[Dict]) -> None:
        """Analyze S3 buckets for optimization"""
        for bucket in resources:
            bucket_name = bucket.get('name', 'unknown')
            
            # Check for lifecycle optimization
            standard_storage_gb = bucket.get('metrics', {}).get('standard_storage_gb', 0)
            last_accessed_days = bucket.get('metrics', {}).get('last_accessed_days', 0)
            
            if standard_storage_gb > 100 and last_accessed_days > 30:
                # Suggest transitioning to Infrequent Access
                monthly_cost = standard_storage_gb * 0.023  # Standard storage cost
                ia_cost = standard_storage_gb * 0.0125  # IA storage cost
                potential_savings = monthly_cost - ia_cost
                
                if potential_savings >= self.min_savings_threshold:
                    self.quick_wins.append(QuickWin(
                        resource_id=bucket_name,
                        resource_type='S3Bucket',
                        region=bucket.get('region', 'unknown'),
                        category='Storage Optimization',
                        title='Transition S3 Objects to Infrequent Access',
                        description=f"Bucket has {standard_storage_gb:.0f}GB not accessed in {last_accessed_days} days",
                        current_monthly_cost=monthly_cost,
                        potential_savings=potential_savings,
                        savings_percentage=(potential_savings / monthly_cost) * 100,
                        implementation_effort=ImplementationEffort.LOW,
                        risk_level=RiskLevel.ZERO,
                        action_steps=[
                            "Create lifecycle policy",
                            "Transition objects older than 30 days to IA",
                            "Consider Glacier for objects older than 90 days"
                        ],
                        confidence=Confidence.HIGH,
                        tags=bucket.get('tags', {}),
                        metadata={
                            'bucket_name': bucket_name,
                            'standard_gb': standard_storage_gb,
                            'last_accessed_days': last_accessed_days
                        }
                    ))
    
    # Helper methods
    def _get_resource_age_days(self, resource: Dict) -> int:
        """Get age of resource in days"""
        created_time = resource.get('create_time', resource.get('start_time', ''))
        if created_time:
            if isinstance(created_time, str):
                created_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            age = datetime.now(created_time.tzinfo) - created_time
            return age.days
        return 0
    
    def _get_state_duration_days(self, resource: Dict, state: str) -> int:
        """Get duration in current state in days"""
        state_transition_time = resource.get('state_transition_time', '')
        if state_transition_time:
            if isinstance(state_transition_time, str):
                state_transition_time = datetime.fromisoformat(state_transition_time.replace('Z', '+00:00'))
            duration = datetime.now(state_transition_time.tzinfo) - state_transition_time
            return duration.days
        return 0
    
    def _calculate_ebs_cost(self, size_gb: int, volume_type: str) -> float:
        """Calculate monthly EBS cost"""
        pricing = {
            'gp3': 0.08,
            'gp2': 0.10,
            'io1': 0.125,
            'io2': 0.125,
            'st1': 0.045,
            'sc1': 0.025
        }
        return size_gb * pricing.get(volume_type, 0.10)
    
    def _estimate_instance_cost(self, instance_type: str) -> float:
        """Estimate monthly EC2 instance cost (simplified)"""
        # Simplified pricing - in production, use AWS Pricing API
        size_multipliers = {
            'nano': 0.5, 'micro': 1, 'small': 2, 'medium': 4,
            'large': 8, 'xlarge': 16, '2xlarge': 32, '4xlarge': 64
        }
        
        for size, multiplier in size_multipliers.items():
            if size in instance_type:
                return multiplier * 5  # Base price $5 for t3.micro equivalent
        
        return 100  # Default for unknown types
    
    def _estimate_rds_cost(self, instance_class: str, storage_gb: int, multi_az: bool) -> float:
        """Estimate monthly RDS cost (simplified)"""
        # Simplified - in production use AWS Pricing API
        base_cost = self._estimate_instance_cost(instance_class) * 1.5  # RDS premium
        storage_cost = storage_gb * 0.115  # GP2 storage
        
        if multi_az:
            base_cost *= 2
        
        return base_cost + storage_cost
    
    def _get_smaller_instance_type(self, instance_type: str) -> Optional[str]:
        """Get next smaller instance type"""
        size_order = ['nano', 'micro', 'small', 'medium', 'large', 'xlarge', '2xlarge', '4xlarge']
        
        for i, size in enumerate(size_order[1:], 1):
            if size in instance_type:
                smaller_size = size_order[i-1]
                return instance_type.replace(size, smaller_size)
        
        return None
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of quick wins"""
        if not self.quick_wins:
            return {
                'total_opportunities': 0,
                'total_monthly_savings': 0,
                'total_annual_savings': 0,
                'by_category': {},
                'by_effort': {},
                'by_risk': {}
            }
        
        total_monthly = sum(qw.potential_savings for qw in self.quick_wins)
        
        by_category = defaultdict(lambda: {'count': 0, 'savings': 0})
        by_effort = defaultdict(lambda: {'count': 0, 'savings': 0})
        by_risk = defaultdict(lambda: {'count': 0, 'savings': 0})
        
        for qw in self.quick_wins:
            by_category[qw.category]['count'] += 1
            by_category[qw.category]['savings'] += qw.potential_savings
            
            by_effort[qw.implementation_effort.value]['count'] += 1
            by_effort[qw.implementation_effort.value]['savings'] += qw.potential_savings
            
            by_risk[qw.risk_level.value]['count'] += 1
            by_risk[qw.risk_level.value]['savings'] += qw.potential_savings
        
        return {
            'total_opportunities': len(self.quick_wins),
            'total_monthly_savings': total_monthly,
            'total_annual_savings': total_monthly * 12,
            'by_category': dict(by_category),
            'by_effort': dict(by_effort),
            'by_risk': dict(by_risk),
            'top_10_opportunities': [
                {
                    'resource_id': qw.resource_id,
                    'title': qw.title,
                    'monthly_savings': qw.potential_savings,
                    'effort': qw.implementation_effort.value,
                    'risk': qw.risk_level.value
                }
                for qw in self.quick_wins[:10]
            ]
        }