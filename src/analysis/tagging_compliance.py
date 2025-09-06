"""
Resource Tagging Analysis and Compliance
Analyzes tag compliance, cost allocation, and governance.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Tag compliance status"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    EXEMPT = "exempt"


class TaggingStrategy(str, Enum):
    """Tagging strategy types"""
    MANDATORY = "mandatory"  # Required for all resources
    RECOMMENDED = "recommended"  # Suggested but not required
    COST_ALLOCATION = "cost_allocation"  # For cost tracking
    AUTOMATION = "automation"  # For automated actions
    SECURITY = "security"  # Security and compliance


@dataclass
class TagPolicy:
    """Defines a tag policy rule"""
    tag_key: str
    tag_values: Optional[List[str]]  # None means any value is acceptable
    strategy: TaggingStrategy
    resource_types: Optional[List[str]]  # None means all resources
    environments: Optional[List[str]]  # None means all environments
    description: str
    enforcement_action: str  # warn, block, auto_tag
    default_value: Optional[str] = None
    regex_pattern: Optional[str] = None
    case_sensitive: bool = False
    
    def validate_value(self, value: str) -> bool:
        """Validate tag value against policy"""
        if not value and self.default_value:
            return True
        
        if self.regex_pattern:
            pattern = re.compile(self.regex_pattern, re.IGNORECASE if not self.case_sensitive else 0)
            if not pattern.match(value):
                return False
        
        if self.tag_values:
            if self.case_sensitive:
                return value in self.tag_values
            else:
                return value.lower() in [v.lower() for v in self.tag_values]
        
        return True


@dataclass
class TagComplianceReport:
    """Tag compliance analysis report"""
    resource_id: str
    resource_type: str
    region: str
    compliance_status: ComplianceStatus
    missing_mandatory_tags: List[str]
    missing_recommended_tags: List[str]
    invalid_tag_values: Dict[str, str]
    existing_tags: Dict[str, str]
    compliance_score: float  # 0-100
    remediation_actions: List[str]
    estimated_cost: float
    last_modified: Optional[datetime] = None
    
    @property
    def is_compliant(self) -> bool:
        return self.compliance_status == ComplianceStatus.COMPLIANT
    
    @property
    def needs_attention(self) -> bool:
        return len(self.missing_mandatory_tags) > 0 or len(self.invalid_tag_values) > 0


@dataclass
class CostAllocationReport:
    """Cost allocation by tags report"""
    tag_key: str
    tag_value: str
    resource_count: int
    monthly_cost: float
    percentage_of_total: float
    trending: str  # increasing, stable, decreasing
    resources: List[str] = field(default_factory=list)
    sub_allocations: Dict[str, float] = field(default_factory=dict)
    
    @property
    def annual_cost(self) -> float:
        return self.monthly_cost * 12


class TaggingComplianceAnalyzer:
    """
    Analyzes resource tagging for compliance and cost allocation.
    Provides governance insights and remediation recommendations.
    """
    
    # Default tag policies
    DEFAULT_POLICIES = [
        TagPolicy(
            tag_key="Environment",
            tag_values=["Production", "Staging", "Development", "Test"],
            strategy=TaggingStrategy.MANDATORY,
            resource_types=None,
            environments=None,
            description="Environment classification",
            enforcement_action="warn",
            case_sensitive=False
        ),
        TagPolicy(
            tag_key="Owner",
            tag_values=None,
            strategy=TaggingStrategy.MANDATORY,
            resource_types=None,
            environments=None,
            description="Resource owner email",
            enforcement_action="warn",
            regex_pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        ),
        TagPolicy(
            tag_key="CostCenter",
            tag_values=None,
            strategy=TaggingStrategy.COST_ALLOCATION,
            resource_types=None,
            environments=["Production"],
            description="Cost center for billing",
            enforcement_action="warn",
            regex_pattern=r"^CC-\d{4}$"
        ),
        TagPolicy(
            tag_key="Project",
            tag_values=None,
            strategy=TaggingStrategy.RECOMMENDED,
            resource_types=None,
            environments=None,
            description="Project name",
            enforcement_action="warn"
        ),
        TagPolicy(
            tag_key="Application",
            tag_values=None,
            strategy=TaggingStrategy.RECOMMENDED,
            resource_types=None,
            environments=None,
            description="Application name",
            enforcement_action="warn"
        ),
        TagPolicy(
            tag_key="DataClassification",
            tag_values=["Public", "Internal", "Confidential", "Restricted"],
            strategy=TaggingStrategy.SECURITY,
            resource_types=["s3_bucket", "rds_instance", "dynamodb_table"],
            environments=None,
            description="Data sensitivity classification",
            enforcement_action="block",
            default_value="Internal"
        ),
        TagPolicy(
            tag_key="BackupPolicy",
            tag_values=["Daily", "Weekly", "Monthly", "None"],
            strategy=TaggingStrategy.AUTOMATION,
            resource_types=["ec2_instance", "rds_instance", "ebs_volume"],
            environments=["Production"],
            description="Backup frequency",
            enforcement_action="auto_tag",
            default_value="Daily"
        ),
        TagPolicy(
            tag_key="AutoShutdown",
            tag_values=["Yes", "No"],
            strategy=TaggingStrategy.AUTOMATION,
            resource_types=["ec2_instance"],
            environments=["Development", "Test"],
            description="Auto-shutdown for cost savings",
            enforcement_action="auto_tag",
            default_value="Yes"
        )
    ]
    
    def __init__(self, policies: Optional[List[TagPolicy]] = None):
        """
        Initialize tagging compliance analyzer.
        
        Args:
            policies: Custom tag policies (uses defaults if None)
        """
        self.policies = policies or self.DEFAULT_POLICIES
        self.compliance_reports = []
        self.cost_allocations = []
        self.tag_statistics = defaultdict(lambda: {
            'usage_count': 0,
            'unique_values': set(),
            'total_cost': 0,
            'resource_types': set()
        })
    
    def analyze_compliance(self, resources: List[Dict]) -> List[TagComplianceReport]:
        """
        Analyze tag compliance for resources.
        
        Args:
            resources: List of resources to analyze
            
        Returns:
            List of compliance reports
        """
        self.compliance_reports = []
        
        for resource in resources:
            report = self._analyze_resource_compliance(resource)
            self.compliance_reports.append(report)
            
            # Update statistics
            self._update_tag_statistics(resource)
        
        return self.compliance_reports
    
    def _analyze_resource_compliance(self, resource: Dict) -> TagComplianceReport:
        """Analyze compliance for a single resource"""
        resource_id = resource.get('id', 'unknown')
        resource_type = resource.get('type', 'unknown')
        region = resource.get('region', 'unknown')
        tags = resource.get('tags', {})
        environment = tags.get('Environment', 'Unknown')
        
        missing_mandatory = []
        missing_recommended = []
        invalid_values = {}
        remediation_actions = []
        
        # Check each policy
        for policy in self.policies:
            # Check if policy applies to this resource
            if not self._policy_applies(policy, resource_type, environment):
                continue
            
            tag_value = tags.get(policy.tag_key)
            
            if not tag_value:
                # Tag is missing
                if policy.strategy == TaggingStrategy.MANDATORY:
                    missing_mandatory.append(policy.tag_key)
                    
                    if policy.default_value:
                        remediation_actions.append(
                            f"Add tag {policy.tag_key}={policy.default_value}"
                        )
                    else:
                        remediation_actions.append(
                            f"Add mandatory tag: {policy.tag_key}"
                        )
                        
                elif policy.strategy == TaggingStrategy.RECOMMENDED:
                    missing_recommended.append(policy.tag_key)
                    remediation_actions.append(
                        f"Consider adding tag: {policy.tag_key}"
                    )
                    
            elif not policy.validate_value(tag_value):
                # Tag value is invalid
                invalid_values[policy.tag_key] = tag_value
                remediation_actions.append(
                    f"Fix invalid value for {policy.tag_key}: '{tag_value}'"
                )
        
        # Calculate compliance score
        total_policies = len([p for p in self.policies 
                            if self._policy_applies(p, resource_type, environment)])
        
        if total_policies > 0:
            violations = len(missing_mandatory) + len(invalid_values)
            compliance_score = max(0, (1 - violations / total_policies)) * 100
        else:
            compliance_score = 100
        
        # Determine compliance status
        if missing_mandatory or invalid_values:
            status = ComplianceStatus.NON_COMPLIANT
        elif missing_recommended:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.COMPLIANT
        
        return TagComplianceReport(
            resource_id=resource_id,
            resource_type=resource_type,
            region=region,
            compliance_status=status,
            missing_mandatory_tags=missing_mandatory,
            missing_recommended_tags=missing_recommended,
            invalid_tag_values=invalid_values,
            existing_tags=tags,
            compliance_score=compliance_score,
            remediation_actions=remediation_actions,
            estimated_cost=resource.get('monthly_cost', 0),
            last_modified=resource.get('last_modified')
        )
    
    def _policy_applies(self, policy: TagPolicy, resource_type: str, environment: str) -> bool:
        """Check if policy applies to resource"""
        # Check resource type
        if policy.resource_types and resource_type not in policy.resource_types:
            return False
        
        # Check environment
        if policy.environments and environment not in policy.environments:
            return False
        
        return True
    
    def _update_tag_statistics(self, resource: Dict):
        """Update tag usage statistics"""
        tags = resource.get('tags', {})
        resource_type = resource.get('type', 'unknown')
        monthly_cost = resource.get('monthly_cost', 0)
        
        for key, value in tags.items():
            self.tag_statistics[key]['usage_count'] += 1
            self.tag_statistics[key]['unique_values'].add(value)
            self.tag_statistics[key]['total_cost'] += monthly_cost
            self.tag_statistics[key]['resource_types'].add(resource_type)
    
    def analyze_cost_allocation(self, resources: List[Dict], 
                               allocation_tags: List[str]) -> List[CostAllocationReport]:
        """
        Analyze cost allocation by tags.
        
        Args:
            resources: List of resources
            allocation_tags: Tag keys to use for cost allocation
            
        Returns:
            List of cost allocation reports
        """
        self.cost_allocations = []
        total_cost = sum(r.get('monthly_cost', 0) for r in resources)
        
        for tag_key in allocation_tags:
            allocations = defaultdict(lambda: {
                'resources': [],
                'cost': 0,
                'sub_allocations': defaultdict(float)
            })
            
            # Group resources by tag value
            for resource in resources:
                tag_value = resource.get('tags', {}).get(tag_key, 'Untagged')
                monthly_cost = resource.get('monthly_cost', 0)
                
                allocations[tag_value]['resources'].append(resource.get('id'))
                allocations[tag_value]['cost'] += monthly_cost
                
                # Track sub-allocations by resource type
                resource_type = resource.get('type', 'unknown')
                allocations[tag_value]['sub_allocations'][resource_type] += monthly_cost
            
            # Create allocation reports
            for tag_value, data in allocations.items():
                if data['cost'] > 0:
                    report = CostAllocationReport(
                        tag_key=tag_key,
                        tag_value=tag_value,
                        resource_count=len(data['resources']),
                        monthly_cost=data['cost'],
                        percentage_of_total=(data['cost'] / total_cost * 100) if total_cost > 0 else 0,
                        trending=self._analyze_cost_trend(tag_key, tag_value),
                        resources=data['resources'][:10],  # Limit to 10 for brevity
                        sub_allocations=dict(data['sub_allocations'])
                    )
                    self.cost_allocations.append(report)
        
        # Sort by cost
        self.cost_allocations.sort(key=lambda x: x.monthly_cost, reverse=True)
        
        return self.cost_allocations
    
    def _analyze_cost_trend(self, tag_key: str, tag_value: str) -> str:
        """Analyze cost trend for tag value"""
        # Simplified - in production, analyze historical data
        return "stable"
    
    def get_untagged_resources(self, resources: List[Dict], 
                              tag_keys: Optional[List[str]] = None) -> List[Dict]:
        """
        Find resources missing specified tags.
        
        Args:
            resources: List of resources
            tag_keys: Specific tags to check (None checks mandatory tags)
            
        Returns:
            List of untagged resources
        """
        if not tag_keys:
            # Use mandatory tags from policies
            tag_keys = [p.tag_key for p in self.policies 
                       if p.strategy == TaggingStrategy.MANDATORY]
        
        untagged = []
        
        for resource in resources:
            tags = resource.get('tags', {})
            missing_tags = [key for key in tag_keys if key not in tags]
            
            if missing_tags:
                untagged.append({
                    'resource_id': resource.get('id'),
                    'resource_type': resource.get('type'),
                    'region': resource.get('region'),
                    'missing_tags': missing_tags,
                    'existing_tags': tags,
                    'monthly_cost': resource.get('monthly_cost', 0)
                })
        
        return untagged
    
    def generate_tagging_recommendations(self, resources: List[Dict]) -> List[Dict]:
        """
        Generate tagging recommendations based on patterns.
        
        Args:
            resources: List of resources
            
        Returns:
            List of tagging recommendations
        """
        recommendations = []
        
        # Analyze tag usage patterns
        tag_usage = self._analyze_tag_usage_patterns(resources)
        
        # Recommend commonly used tags
        for tag_key, usage_info in tag_usage.items():
            if usage_info['coverage'] < 50 and usage_info['coverage'] > 10:
                recommendations.append({
                    'type': 'expand_tag_usage',
                    'tag_key': tag_key,
                    'description': f"Tag '{tag_key}' is used on {usage_info['coverage']:.1f}% of resources",
                    'recommendation': f"Consider applying '{tag_key}' tag to remaining resources",
                    'affected_resources': usage_info['untagged_count'],
                    'potential_benefit': 'Improved cost allocation and governance'
                })
        
        # Recommend standardization
        for tag_key in self.tag_statistics:
            unique_values = self.tag_statistics[tag_key]['unique_values']
            if len(unique_values) > 20:
                recommendations.append({
                    'type': 'standardize_values',
                    'tag_key': tag_key,
                    'description': f"Tag '{tag_key}' has {len(unique_values)} unique values",
                    'recommendation': 'Standardize tag values for better grouping',
                    'current_values': list(unique_values)[:10],  # Sample
                    'potential_benefit': 'Simplified reporting and automation'
                })
        
        # Recommend cost allocation tags
        high_cost_untagged = [r for r in resources 
                             if r.get('monthly_cost', 0) > 100 
                             and 'CostCenter' not in r.get('tags', {})]
        
        if high_cost_untagged:
            recommendations.append({
                'type': 'add_cost_allocation',
                'tag_key': 'CostCenter',
                'description': f"{len(high_cost_untagged)} high-cost resources lack cost allocation tags",
                'recommendation': 'Add CostCenter tags for accurate cost tracking',
                'affected_resources': len(high_cost_untagged),
                'monthly_cost': sum(r.get('monthly_cost', 0) for r in high_cost_untagged),
                'potential_benefit': 'Accurate departmental cost allocation'
            })
        
        # Recommend automation tags
        dev_resources = [r for r in resources 
                        if r.get('tags', {}).get('Environment') in ['Development', 'Test']]
        
        missing_automation = [r for r in dev_resources 
                            if 'AutoShutdown' not in r.get('tags', {})]
        
        if missing_automation:
            potential_savings = sum(r.get('monthly_cost', 0) for r in missing_automation) * 0.3
            recommendations.append({
                'type': 'add_automation_tags',
                'tag_key': 'AutoShutdown',
                'description': f"{len(missing_automation)} dev/test resources lack automation tags",
                'recommendation': 'Add AutoShutdown tags to enable cost-saving automation',
                'affected_resources': len(missing_automation),
                'potential_savings': potential_savings,
                'potential_benefit': 'Automated cost savings in non-production'
            })
        
        return recommendations
    
    def _analyze_tag_usage_patterns(self, resources: List[Dict]) -> Dict[str, Dict]:
        """Analyze tag usage patterns across resources"""
        patterns = {}
        total_resources = len(resources)
        
        if total_resources == 0:
            return patterns
        
        # Count tag usage
        tag_counts = defaultdict(int)
        for resource in resources:
            for tag_key in resource.get('tags', {}):
                tag_counts[tag_key] += 1
        
        # Calculate coverage
        for tag_key, count in tag_counts.items():
            patterns[tag_key] = {
                'coverage': (count / total_resources) * 100,
                'tagged_count': count,
                'untagged_count': total_resources - count
            }
        
        return patterns
    
    def create_tag_policy(self, tag_key: str, **kwargs) -> TagPolicy:
        """
        Create a new tag policy.
        
        Args:
            tag_key: Tag key name
            **kwargs: Policy configuration
            
        Returns:
            New TagPolicy object
        """
        policy = TagPolicy(
            tag_key=tag_key,
            tag_values=kwargs.get('tag_values'),
            strategy=TaggingStrategy(kwargs.get('strategy', 'recommended')),
            resource_types=kwargs.get('resource_types'),
            environments=kwargs.get('environments'),
            description=kwargs.get('description', f"Policy for {tag_key}"),
            enforcement_action=kwargs.get('enforcement_action', 'warn'),
            default_value=kwargs.get('default_value'),
            regex_pattern=kwargs.get('regex_pattern'),
            case_sensitive=kwargs.get('case_sensitive', False)
        )
        
        self.policies.append(policy)
        return policy
    
    def export_compliance_report(self) -> Dict[str, Any]:
        """Export comprehensive compliance report"""
        if not self.compliance_reports:
            return {}
        
        total_resources = len(self.compliance_reports)
        compliant = sum(1 for r in self.compliance_reports if r.is_compliant)
        non_compliant = sum(1 for r in self.compliance_reports 
                          if r.compliance_status == ComplianceStatus.NON_COMPLIANT)
        
        # Group by compliance status
        by_status = defaultdict(list)
        for report in self.compliance_reports:
            by_status[report.compliance_status.value].append(report.resource_id)
        
        # Calculate costs
        compliant_cost = sum(r.estimated_cost for r in self.compliance_reports if r.is_compliant)
        non_compliant_cost = sum(r.estimated_cost for r in self.compliance_reports 
                               if r.compliance_status == ComplianceStatus.NON_COMPLIANT)
        
        # Find most common missing tags
        missing_tags_count = defaultdict(int)
        for report in self.compliance_reports:
            for tag in report.missing_mandatory_tags:
                missing_tags_count[tag] += 1
        
        return {
            'summary': {
                'total_resources': total_resources,
                'compliant': compliant,
                'non_compliant': non_compliant,
                'compliance_percentage': (compliant / total_resources * 100) if total_resources > 0 else 0,
                'average_compliance_score': sum(r.compliance_score for r in self.compliance_reports) / total_resources if total_resources > 0 else 0
            },
            'by_status': dict(by_status),
            'costs': {
                'compliant_resources_cost': compliant_cost,
                'non_compliant_resources_cost': non_compliant_cost,
                'total_monthly_cost': compliant_cost + non_compliant_cost
            },
            'common_issues': {
                'most_missing_tags': sorted(missing_tags_count.items(), key=lambda x: x[1], reverse=True)[:5],
                'resources_needing_attention': sum(1 for r in self.compliance_reports if r.needs_attention)
            },
            'tag_statistics': {
                tag: {
                    'usage_count': stats['usage_count'],
                    'unique_values': len(stats['unique_values']),
                    'total_cost': stats['total_cost'],
                    'resource_types': len(stats['resource_types'])
                }
                for tag, stats in self.tag_statistics.items()
            },
            'policy_summary': {
                'total_policies': len(self.policies),
                'mandatory_policies': sum(1 for p in self.policies if p.strategy == TaggingStrategy.MANDATORY),
                'cost_allocation_policies': sum(1 for p in self.policies if p.strategy == TaggingStrategy.COST_ALLOCATION),
                'automation_policies': sum(1 for p in self.policies if p.strategy == TaggingStrategy.AUTOMATION)
            }
        }