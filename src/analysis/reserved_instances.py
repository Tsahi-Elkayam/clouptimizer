"""
Reserved Instance & Savings Plans Analyzer
Analyzes RI utilization and recommends optimal purchase strategies.
Provides recommendations for new purchases, modifications, and Savings Plans.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, date
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class PaymentOption(str, Enum):
    """RI/Savings Plan payment options"""
    NO_UPFRONT = "no_upfront"
    PARTIAL_UPFRONT = "partial_upfront"
    ALL_UPFRONT = "all_upfront"


class RecommendationType(str, Enum):
    """Types of RI/Savings Plan recommendations"""
    RI_PURCHASE = "ri_purchase"
    RI_MODIFY = "ri_modify"
    RI_EXCHANGE = "ri_exchange"
    SAVINGS_PLAN_PURCHASE = "savings_plan_purchase"
    TERMINATE_UNUSED = "terminate_unused"
    CONVERTIBLE_UPGRADE = "convertible_upgrade"


class InstanceFlexibility(str, Enum):
    """Instance size flexibility options"""
    LINUX = "linux"  # Size flexible
    WINDOWS = "windows"  # Not size flexible
    RHEL = "rhel"  # Not size flexible
    SUSE = "suse"  # Size flexible


@dataclass
class ReservationRecommendation:
    """Reserved Instance or Savings Plan recommendation"""
    recommendation_id: str
    recommendation_type: RecommendationType
    service: str  # ec2, rds, elasticache, etc.
    instance_family: str
    instance_type: Optional[str]  # Specific type for RIs
    region: str
    platform: str  # Linux, Windows, etc.
    current_on_demand_cost: float
    current_on_demand_hours: float
    recommended_quantity: int
    reserved_cost: float
    monthly_savings: float
    annual_savings: float
    upfront_cost: float
    payback_period_months: float
    term_length: int  # 1 or 3 years
    payment_option: PaymentOption
    confidence_level: str  # HIGH, MEDIUM, LOW
    risk_assessment: str
    break_even_months: int
    implementation_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def roi_percentage(self) -> float:
        """Calculate return on investment"""
        if self.upfront_cost > 0:
            return (self.annual_savings / self.upfront_cost) * 100
        return float('inf')  # No upfront = infinite ROI
    
    @property
    def effective_discount(self) -> float:
        """Calculate effective discount vs on-demand"""
        if self.current_on_demand_cost > 0:
            return ((self.current_on_demand_cost - self.reserved_cost) / 
                   self.current_on_demand_cost) * 100
        return 0


@dataclass
class CurrentReservation:
    """Current Reserved Instance or Savings Plan"""
    reservation_id: str
    reservation_type: str  # ec2_ri, rds_ri, compute_savings_plan, etc.
    instance_type: Optional[str]
    instance_family: Optional[str]
    region: str
    platform: str
    quantity: int
    utilization_percentage: float
    monthly_cost: float
    hourly_cost: float
    remaining_term_months: int
    expiration_date: date
    payment_option: PaymentOption
    modification_eligible: bool
    exchange_eligible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_underutilized(self) -> bool:
        """Check if reservation is underutilized"""
        return self.utilization_percentage < 70
    
    @property
    def is_expiring_soon(self) -> bool:
        """Check if reservation expires within 90 days"""
        return self.remaining_term_months <= 3


@dataclass
class UsagePattern:
    """Resource usage pattern for RI analysis"""
    resource_id: str
    instance_type: str
    instance_family: str
    region: str
    platform: str
    hourly_usage: List[float]  # 24 hours of usage data
    daily_usage: List[float]  # 30 days of usage data
    monthly_usage: List[float]  # 12 months of usage data
    average_hours_per_day: float
    peak_hours_per_day: float
    minimum_hours_per_day: float
    consistency_score: float  # 0-100, how consistent is usage
    
    @property
    def is_steady_state(self) -> bool:
        """Check if usage is steady state (good for RI)"""
        return self.consistency_score > 70 and self.minimum_hours_per_day >= 16
    
    @property
    def is_variable(self) -> bool:
        """Check if usage is variable (better for Savings Plans)"""
        return self.consistency_score < 70 or self.minimum_hours_per_day < 16


class ReservedInstanceAnalyzer:
    """
    Analyzes Reserved Instances and Savings Plans for optimization opportunities.
    Provides purchase recommendations, modification suggestions, and utilization analysis.
    """
    
    # Approximate RI discount rates by payment option and term
    RI_DISCOUNT_RATES = {
        'ec2': {
            'linux': {
                'no_upfront_1yr': 0.31,     # ~31% discount vs on-demand
                'partial_upfront_1yr': 0.34, # ~34% discount
                'all_upfront_1yr': 0.36,     # ~36% discount
                'no_upfront_3yr': 0.53,      # ~53% discount
                'partial_upfront_3yr': 0.56, # ~56% discount
                'all_upfront_3yr': 0.59,     # ~59% discount
            },
            'windows': {
                'no_upfront_1yr': 0.20,
                'partial_upfront_1yr': 0.23,
                'all_upfront_1yr': 0.25,
                'no_upfront_3yr': 0.42,
                'partial_upfront_3yr': 0.45,
                'all_upfront_3yr': 0.48,
            }
        },
        'rds': {
            'mysql': {
                'no_upfront_1yr': 0.28,
                'partial_upfront_1yr': 0.31,
                'all_upfront_1yr': 0.33,
                'no_upfront_3yr': 0.48,
                'partial_upfront_3yr': 0.51,
                'all_upfront_3yr': 0.54,
            },
            'postgresql': {
                'no_upfront_1yr': 0.28,
                'partial_upfront_1yr': 0.31,
                'all_upfront_1yr': 0.33,
                'no_upfront_3yr': 0.48,
                'partial_upfront_3yr': 0.51,
                'all_upfront_3yr': 0.54,
            }
        },
        'elasticache': {
            'redis': {
                'no_upfront_1yr': 0.25,
                'partial_upfront_1yr': 0.28,
                'all_upfront_1yr': 0.30,
                'no_upfront_3yr': 0.45,
                'partial_upfront_3yr': 0.48,
                'all_upfront_3yr': 0.51,
            }
        }
    }
    
    # Savings Plans discount rates
    SAVINGS_PLAN_RATES = {
        'compute': {
            'no_upfront_1yr': 0.20,
            'partial_upfront_1yr': 0.23,
            'all_upfront_1yr': 0.25,
            'no_upfront_3yr': 0.37,
            'partial_upfront_3yr': 0.40,
            'all_upfront_3yr': 0.43,
        },
        'ec2_instance': {
            'no_upfront_1yr': 0.28,
            'partial_upfront_1yr': 0.31,
            'all_upfront_1yr': 0.33,
            'no_upfront_3yr': 0.49,
            'partial_upfront_3yr': 0.52,
            'all_upfront_3yr': 0.55,
        }
    }
    
    # Thresholds for recommendations
    THRESHOLDS = {
        'min_utilization_hours': 500,  # Minimum hours/month to recommend RI
        'min_savings_monthly': 50,     # Minimum $50/month savings
        'utilization_target': 85,      # Target utilization percentage
        'consistency_threshold': 70,   # Minimum consistency score for RI
        'break_even_months': 9,        # Maximum break-even period
    }
    
    def __init__(self, min_savings_threshold: float = 50.0):
        """
        Initialize Reserved Instance Analyzer.
        
        Args:
            min_savings_threshold: Minimum monthly savings to recommend
        """
        self.min_savings_threshold = min_savings_threshold
        self.current_reservations: List[CurrentReservation] = []
        self.usage_patterns: List[UsagePattern] = []
        self.recommendations: List[ReservationRecommendation] = []
        self._recommendation_counter = 0
    
    def analyze_current_reservations(self, reservations: List[Dict]) -> List[CurrentReservation]:
        """
        Analyze current Reserved Instances and Savings Plans.
        
        Args:
            reservations: List of current reservations
            
        Returns:
            List of CurrentReservation objects
        """
        self.current_reservations = []
        
        for res in reservations:
            current = CurrentReservation(
                reservation_id=res.get('reservation_id', 'unknown'),
                reservation_type=res.get('type', 'ec2_ri'),
                instance_type=res.get('instance_type'),
                instance_family=res.get('instance_family'),
                region=res.get('region', 'us-east-1'),
                platform=res.get('platform', 'linux'),
                quantity=res.get('quantity', 1),
                utilization_percentage=res.get('utilization', 0),
                monthly_cost=res.get('monthly_cost', 0),
                hourly_cost=res.get('hourly_cost', 0),
                remaining_term_months=self._calculate_remaining_months(res.get('end_date')),
                expiration_date=self._parse_date(res.get('end_date')),
                payment_option=PaymentOption(res.get('payment_option', 'no_upfront')),
                modification_eligible=res.get('modification_eligible', False),
                exchange_eligible=res.get('exchange_eligible', False),
                metadata=res.get('metadata', {})
            )
            
            self.current_reservations.append(current)
            
            # Check for optimization opportunities
            if current.is_underutilized:
                self._analyze_underutilized_reservation(current)
            
            if current.is_expiring_soon:
                self._analyze_expiring_reservation(current)
        
        return self.current_reservations
    
    def analyze_usage_patterns(self, usage_data: List[Dict]) -> List[UsagePattern]:
        """
        Analyze usage patterns to identify RI opportunities.
        
        Args:
            usage_data: Historical usage data
            
        Returns:
            List of UsagePattern objects
        """
        self.usage_patterns = []
        
        # Group usage by instance family and region
        usage_by_family = defaultdict(lambda: defaultdict(list))
        
        for usage in usage_data:
            family = self._get_instance_family(usage.get('instance_type', ''))
            region = usage.get('region', 'us-east-1')
            key = f"{family}:{region}"
            usage_by_family[key]['hours'].append(usage.get('hours', 0))
            usage_by_family[key]['cost'].append(usage.get('cost', 0))
            usage_by_family[key]['instances'].append(usage)
        
        # Analyze each family/region combination
        for key, data in usage_by_family.items():
            family, region = key.split(':')
            
            pattern = UsagePattern(
                resource_id=f"pattern-{family}-{region}",
                instance_type=self._get_common_instance_type(data['instances']),
                instance_family=family,
                region=region,
                platform=self._get_common_platform(data['instances']),
                hourly_usage=self._calculate_hourly_pattern(data['hours']),
                daily_usage=self._calculate_daily_pattern(data['hours']),
                monthly_usage=self._calculate_monthly_pattern(data['hours']),
                average_hours_per_day=statistics.mean(data['hours']) if data['hours'] else 0,
                peak_hours_per_day=max(data['hours']) if data['hours'] else 0,
                minimum_hours_per_day=min(data['hours']) if data['hours'] else 0,
                consistency_score=self._calculate_consistency_score(data['hours'])
            )
            
            self.usage_patterns.append(pattern)
            
            # Generate recommendations based on pattern
            if pattern.is_steady_state:
                self._recommend_reserved_instance(pattern, data)
            elif pattern.is_variable:
                self._recommend_savings_plan(pattern, data)
        
        return self.usage_patterns
    
    def generate_purchase_recommendations(self) -> List[ReservationRecommendation]:
        """
        Generate RI and Savings Plan purchase recommendations.
        
        Returns:
            List of purchase recommendations
        """
        recommendations = []
        
        # Analyze coverage gaps
        coverage_gaps = self._identify_coverage_gaps()
        
        for gap in coverage_gaps:
            if gap['steady_hours'] > self.THRESHOLDS['min_utilization_hours']:
                # Recommend RI for steady-state workloads
                rec = self._create_ri_recommendation(gap)
                if rec and rec.monthly_savings >= self.min_savings_threshold:
                    recommendations.append(rec)
            
            elif gap['variable_hours'] > 300:  # Significant variable usage
                # Recommend Savings Plans for variable workloads
                rec = self._create_savings_plan_recommendation(gap)
                if rec and rec.monthly_savings >= self.min_savings_threshold:
                    recommendations.append(rec)
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.annual_savings, reverse=True)
        self.recommendations.extend(recommendations)
        
        return recommendations
    
    def analyze_modification_opportunities(self) -> List[ReservationRecommendation]:
        """
        Analyze opportunities to modify existing reservations.
        
        Returns:
            List of modification recommendations
        """
        modifications = []
        
        for reservation in self.current_reservations:
            if not reservation.modification_eligible:
                continue
            
            # Check if instance type should be changed
            current_usage = self._get_current_usage_for_reservation(reservation)
            
            if current_usage:
                optimal_type = self._find_optimal_instance_type(
                    current_usage,
                    reservation.instance_family
                )
                
                if optimal_type != reservation.instance_type:
                    rec = self._create_modification_recommendation(
                        reservation,
                        optimal_type,
                        current_usage
                    )
                    
                    if rec and rec.monthly_savings >= self.min_savings_threshold:
                        modifications.append(rec)
        
        self.recommendations.extend(modifications)
        return modifications
    
    def calculate_optimal_coverage(self, usage_history: List[Dict],
                                  target_coverage: float = 0.7) -> Dict[str, Any]:
        """
        Calculate optimal RI/Savings Plan coverage.
        
        Args:
            usage_history: Historical usage data
            target_coverage: Target coverage percentage (default 70%)
            
        Returns:
            Optimal coverage recommendations
        """
        # Analyze usage distribution
        usage_percentiles = self._calculate_usage_percentiles(usage_history)
        
        # Determine baseline (steady-state) usage
        baseline_usage = usage_percentiles['p30']  # 30th percentile as baseline
        
        # Calculate coverage recommendations
        coverage_plan = {
            'target_coverage_percent': target_coverage * 100,
            'baseline_hours': baseline_usage,
            'recommended_ris': [],
            'recommended_savings_plans': [],
            'estimated_coverage': 0,
            'estimated_monthly_savings': 0,
            'estimated_annual_savings': 0
        }
        
        # Recommend RIs for baseline
        if baseline_usage > self.THRESHOLDS['min_utilization_hours']:
            ri_recommendations = self._calculate_ri_coverage(
                baseline_usage,
                usage_history
            )
            coverage_plan['recommended_ris'] = ri_recommendations
        
        # Recommend Savings Plans for variable usage above baseline
        variable_usage = usage_percentiles['p70'] - baseline_usage
        if variable_usage > 100:  # Significant variable usage
            sp_recommendations = self._calculate_sp_coverage(
                variable_usage,
                usage_history
            )
            coverage_plan['recommended_savings_plans'] = sp_recommendations
        
        # Calculate total savings
        total_savings = sum(r['monthly_savings'] for r in 
                          coverage_plan['recommended_ris'] + 
                          coverage_plan['recommended_savings_plans'])
        
        coverage_plan['estimated_monthly_savings'] = total_savings
        coverage_plan['estimated_annual_savings'] = total_savings * 12
        
        # Calculate coverage percentage
        total_hours = sum(u.get('hours', 0) for u in usage_history)
        covered_hours = (baseline_usage * len(coverage_plan['recommended_ris']) +
                        variable_usage * len(coverage_plan['recommended_savings_plans']))
        
        coverage_plan['estimated_coverage'] = (covered_hours / total_hours * 100 
                                              if total_hours > 0 else 0)
        
        return coverage_plan
    
    def _analyze_underutilized_reservation(self, reservation: CurrentReservation) -> None:
        """Analyze underutilized reservation for optimization"""
        if reservation.utilization_percentage < 50:
            # Severe underutilization - recommend exchange or termination
            if reservation.exchange_eligible:
                self._recommend_exchange(reservation)
            else:
                self._recommend_reallocation(reservation)
        
        elif reservation.utilization_percentage < 70:
            # Moderate underutilization - recommend modification
            if reservation.modification_eligible:
                self._recommend_modification(reservation)
    
    def _analyze_expiring_reservation(self, reservation: CurrentReservation) -> None:
        """Analyze expiring reservation for renewal"""
        # Get recent utilization
        if reservation.utilization_percentage > 80:
            # High utilization - recommend renewal
            self._recommend_renewal(reservation)
        elif reservation.utilization_percentage > 50:
            # Moderate utilization - recommend adjusted renewal
            self._recommend_adjusted_renewal(reservation)
        # Low utilization - no renewal recommendation
    
    def _recommend_reserved_instance(self, pattern: UsagePattern, data: Dict) -> None:
        """Generate RI recommendation for steady-state usage"""
        # Calculate costs for different options
        on_demand_cost = self._calculate_on_demand_cost(pattern)
        
        best_option = None
        best_savings = 0
        
        for term in [1, 3]:  # 1-year and 3-year terms
            for payment in PaymentOption:
                ri_cost, upfront = self._calculate_ri_cost(
                    pattern,
                    term,
                    payment
                )
                
                monthly_savings = on_demand_cost - ri_cost
                
                if monthly_savings > best_savings:
                    best_savings = monthly_savings
                    best_option = {
                        'term': term,
                        'payment': payment,
                        'ri_cost': ri_cost,
                        'upfront': upfront,
                        'monthly_savings': monthly_savings
                    }
        
        if best_option and best_savings >= self.min_savings_threshold:
            recommendation = ReservationRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                recommendation_type=RecommendationType.RI_PURCHASE,
                service='ec2',
                instance_family=pattern.instance_family,
                instance_type=pattern.instance_type,
                region=pattern.region,
                platform=pattern.platform,
                current_on_demand_cost=on_demand_cost,
                current_on_demand_hours=pattern.average_hours_per_day * 30,
                recommended_quantity=1,
                reserved_cost=best_option['ri_cost'],
                monthly_savings=best_option['monthly_savings'],
                annual_savings=best_option['monthly_savings'] * 12,
                upfront_cost=best_option['upfront'],
                payback_period_months=self._calculate_payback_period(
                    best_option['upfront'],
                    best_option['monthly_savings']
                ),
                term_length=best_option['term'],
                payment_option=best_option['payment'],
                confidence_level=self._assess_confidence(pattern),
                risk_assessment=self._assess_risk(pattern),
                break_even_months=self._calculate_break_even(
                    best_option['upfront'],
                    best_option['monthly_savings']
                ),
                implementation_steps=[
                    f"Purchase {best_option['term']}-year Reserved Instance",
                    f"Instance type: {pattern.instance_type}",
                    f"Payment option: {best_option['payment'].value}",
                    f"Region: {pattern.region}",
                    "Monitor utilization post-purchase"
                ]
            )
            
            self.recommendations.append(recommendation)
    
    def _recommend_savings_plan(self, pattern: UsagePattern, data: Dict) -> None:
        """Generate Savings Plan recommendation for variable usage"""
        on_demand_cost = self._calculate_on_demand_cost(pattern)
        
        # Recommend Compute Savings Plan for maximum flexibility
        best_option = None
        best_savings = 0
        
        for term in [1, 3]:
            for payment in PaymentOption:
                sp_cost, upfront = self._calculate_sp_cost(
                    pattern,
                    term,
                    payment,
                    'compute'
                )
                
                monthly_savings = on_demand_cost - sp_cost
                
                if monthly_savings > best_savings:
                    best_savings = monthly_savings
                    best_option = {
                        'term': term,
                        'payment': payment,
                        'sp_cost': sp_cost,
                        'upfront': upfront,
                        'monthly_savings': monthly_savings
                    }
        
        if best_option and best_savings >= self.min_savings_threshold:
            recommendation = ReservationRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                recommendation_type=RecommendationType.SAVINGS_PLAN_PURCHASE,
                service='compute',
                instance_family='flexible',
                instance_type=None,
                region='global',
                platform='all',
                current_on_demand_cost=on_demand_cost,
                current_on_demand_hours=pattern.average_hours_per_day * 30,
                recommended_quantity=1,
                reserved_cost=best_option['sp_cost'],
                monthly_savings=best_option['monthly_savings'],
                annual_savings=best_option['monthly_savings'] * 12,
                upfront_cost=best_option['upfront'],
                payback_period_months=self._calculate_payback_period(
                    best_option['upfront'],
                    best_option['monthly_savings']
                ),
                term_length=best_option['term'],
                payment_option=best_option['payment'],
                confidence_level="HIGH",
                risk_assessment="LOW",
                break_even_months=self._calculate_break_even(
                    best_option['upfront'],
                    best_option['monthly_savings']
                ),
                implementation_steps=[
                    f"Purchase {best_option['term']}-year Compute Savings Plan",
                    f"Commitment: ${on_demand_cost * 0.7:.2f}/hour",
                    f"Payment option: {best_option['payment'].value}",
                    "Applies to EC2, Fargate, and Lambda",
                    "Monitor coverage and utilization"
                ]
            )
            
            self.recommendations.append(recommendation)
    
    # Helper methods
    def _identify_coverage_gaps(self) -> List[Dict]:
        """Identify gaps in current RI/SP coverage"""
        gaps = []
        
        # Compare usage patterns with current reservations
        for pattern in self.usage_patterns:
            covered_hours = self._get_covered_hours(pattern)
            uncovered_hours = pattern.average_hours_per_day * 30 - covered_hours
            
            if uncovered_hours > self.THRESHOLDS['min_utilization_hours']:
                gaps.append({
                    'instance_family': pattern.instance_family,
                    'region': pattern.region,
                    'platform': pattern.platform,
                    'uncovered_hours': uncovered_hours,
                    'steady_hours': pattern.minimum_hours_per_day * 30,
                    'variable_hours': uncovered_hours - (pattern.minimum_hours_per_day * 30),
                    'monthly_cost': self._calculate_on_demand_cost(pattern)
                })
        
        return gaps
    
    def _get_covered_hours(self, pattern: UsagePattern) -> float:
        """Get hours covered by existing reservations"""
        covered = 0
        
        for reservation in self.current_reservations:
            if (reservation.instance_family == pattern.instance_family and
                reservation.region == pattern.region and
                reservation.platform == pattern.platform):
                covered += reservation.quantity * 730  # Hours per month
        
        return covered
    
    def _calculate_on_demand_cost(self, pattern: UsagePattern) -> float:
        """Calculate on-demand cost for usage pattern"""
        # Simplified - use actual pricing API in production
        hourly_rate = self._get_on_demand_price(
            pattern.instance_type,
            pattern.region,
            pattern.platform
        )
        
        return hourly_rate * pattern.average_hours_per_day * 30
    
    def _calculate_ri_cost(self, pattern: UsagePattern, term: int,
                          payment: PaymentOption) -> Tuple[float, float]:
        """Calculate RI cost and upfront payment"""
        platform_key = pattern.platform.lower()
        if platform_key not in self.RI_DISCOUNT_RATES['ec2']:
            platform_key = 'linux'  # Default
        
        discount_key = f"{payment.value}_{term}yr"
        discount_rate = self.RI_DISCOUNT_RATES['ec2'][platform_key].get(discount_key, 0.3)
        
        on_demand_cost = self._calculate_on_demand_cost(pattern)
        ri_monthly_cost = on_demand_cost * (1 - discount_rate)
        
        # Calculate upfront based on payment option
        if payment == PaymentOption.ALL_UPFRONT:
            upfront = ri_monthly_cost * 12 * term
            monthly = 0
        elif payment == PaymentOption.PARTIAL_UPFRONT:
            upfront = ri_monthly_cost * 6 * term  # ~50% upfront
            monthly = ri_monthly_cost * 0.5
        else:  # NO_UPFRONT
            upfront = 0
            monthly = ri_monthly_cost
        
        return monthly, upfront
    
    def _calculate_sp_cost(self, pattern: UsagePattern, term: int,
                          payment: PaymentOption, sp_type: str) -> Tuple[float, float]:
        """Calculate Savings Plan cost and upfront payment"""
        discount_key = f"{payment.value}_{term}yr"
        discount_rate = self.SAVINGS_PLAN_RATES[sp_type].get(discount_key, 0.2)
        
        on_demand_cost = self._calculate_on_demand_cost(pattern)
        sp_monthly_cost = on_demand_cost * (1 - discount_rate)
        
        # Calculate upfront based on payment option
        if payment == PaymentOption.ALL_UPFRONT:
            upfront = sp_monthly_cost * 12 * term
            monthly = 0
        elif payment == PaymentOption.PARTIAL_UPFRONT:
            upfront = sp_monthly_cost * 6 * term
            monthly = sp_monthly_cost * 0.5
        else:  # NO_UPFRONT
            upfront = 0
            monthly = sp_monthly_cost
        
        return monthly, upfront
    
    def _get_on_demand_price(self, instance_type: str, region: str,
                            platform: str) -> float:
        """Get on-demand hourly price"""
        # Simplified pricing - integrate with AWS Pricing API
        base_prices = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'r5.large': 0.126,
            'r5.xlarge': 0.252,
        }
        
        price = base_prices.get(instance_type, 0.1)
        
        # Adjust for platform
        if platform.lower() == 'windows':
            price *= 1.5
        elif platform.lower() in ['rhel', 'suse']:
            price *= 1.2
        
        # Adjust for region (simplified)
        if 'ap-' in region or 'sa-' in region:
            price *= 1.2
        
        return price
    
    def _calculate_payback_period(self, upfront: float, monthly_savings: float) -> float:
        """Calculate payback period in months"""
        if monthly_savings <= 0:
            return float('inf')
        if upfront == 0:
            return 0
        return upfront / monthly_savings
    
    def _calculate_break_even(self, upfront: float, monthly_savings: float) -> int:
        """Calculate break-even point in months"""
        if monthly_savings <= 0:
            return 999
        if upfront == 0:
            return 0
        return int(upfront / monthly_savings) + 1
    
    def _assess_confidence(self, pattern: UsagePattern) -> str:
        """Assess confidence level for recommendation"""
        if pattern.consistency_score > 85 and pattern.minimum_hours_per_day > 20:
            return "HIGH"
        elif pattern.consistency_score > 70 and pattern.minimum_hours_per_day > 12:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_risk(self, pattern: UsagePattern) -> str:
        """Assess risk level for recommendation"""
        if pattern.consistency_score > 85:
            return "LOW"
        elif pattern.consistency_score > 70:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_consistency_score(self, usage_hours: List[float]) -> float:
        """Calculate usage consistency score"""
        if not usage_hours or len(usage_hours) < 2:
            return 0
        
        mean = statistics.mean(usage_hours)
        if mean == 0:
            return 0
        
        std_dev = statistics.stdev(usage_hours)
        coefficient_of_variation = (std_dev / mean) * 100
        
        # Lower CV means more consistent
        consistency = max(0, 100 - coefficient_of_variation)
        return min(100, consistency)
    
    def _get_instance_family(self, instance_type: str) -> str:
        """Extract instance family from type"""
        if not instance_type:
            return "unknown"
        
        # Extract family (e.g., 'm5' from 'm5.large')
        parts = instance_type.split('.')
        if parts:
            return parts[0]
        return instance_type
    
    def _calculate_remaining_months(self, end_date: Any) -> int:
        """Calculate remaining months until expiration"""
        if not end_date:
            return 0
        
        end = self._parse_date(end_date)
        today = date.today()
        
        months = (end.year - today.year) * 12 + end.month - today.month
        return max(0, months)
    
    def _parse_date(self, date_str: Any) -> date:
        """Parse date from various formats"""
        if isinstance(date_str, date):
            return date_str
        
        if isinstance(date_str, str):
            try:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
            except:
                return date.today()
        
        return date.today()
    
    def _generate_recommendation_id(self) -> str:
        """Generate unique recommendation ID"""
        self._recommendation_counter += 1
        return f"ri-rec-{self._recommendation_counter:04d}"
    
    def _get_common_instance_type(self, instances: List[Dict]) -> str:
        """Get most common instance type"""
        if not instances:
            return "unknown"
        
        types = [i.get('instance_type', '') for i in instances]
        if types:
            return max(set(types), key=types.count)
        return "unknown"
    
    def _get_common_platform(self, instances: List[Dict]) -> str:
        """Get most common platform"""
        if not instances:
            return "linux"
        
        platforms = [i.get('platform', 'linux') for i in instances]
        if platforms:
            return max(set(platforms), key=platforms.count)
        return "linux"
    
    def _calculate_hourly_pattern(self, hours_data: List[float]) -> List[float]:
        """Calculate hourly usage pattern"""
        # Simplified - return 24 hours of average usage
        if not hours_data:
            return [0] * 24
        
        avg = statistics.mean(hours_data)
        # Simulate daily pattern (higher during business hours)
        pattern = []
        for hour in range(24):
            if 8 <= hour <= 18:  # Business hours
                pattern.append(avg * 1.5)
            else:
                pattern.append(avg * 0.5)
        
        return pattern
    
    def _calculate_daily_pattern(self, hours_data: List[float]) -> List[float]:
        """Calculate daily usage pattern"""
        # Return last 30 days or pad with average
        if len(hours_data) >= 30:
            return hours_data[-30:]
        else:
            avg = statistics.mean(hours_data) if hours_data else 0
            return hours_data + [avg] * (30 - len(hours_data))
    
    def _calculate_monthly_pattern(self, hours_data: List[float]) -> List[float]:
        """Calculate monthly usage pattern"""
        # Return last 12 months or pad with average
        if len(hours_data) >= 12:
            return hours_data[-12:]
        else:
            avg = statistics.mean(hours_data) if hours_data else 0
            return hours_data + [avg] * (12 - len(hours_data))
    
    def _calculate_usage_percentiles(self, usage_history: List[Dict]) -> Dict[str, float]:
        """Calculate usage percentiles"""
        hours = [u.get('hours', 0) for u in usage_history]
        if not hours:
            return {'p30': 0, 'p50': 0, 'p70': 0, 'p90': 0}
        
        sorted_hours = sorted(hours)
        n = len(sorted_hours)
        
        return {
            'p30': sorted_hours[int(n * 0.3)],
            'p50': sorted_hours[int(n * 0.5)],
            'p70': sorted_hours[int(n * 0.7)],
            'p90': sorted_hours[int(n * 0.9)]
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of RI/SP analysis"""
        total_current_ris = len(self.current_reservations)
        total_recommendations = len(self.recommendations)
        
        if not self.recommendations:
            return {
                'current_reservations': total_current_ris,
                'total_recommendations': 0,
                'potential_monthly_savings': 0,
                'potential_annual_savings': 0
            }
        
        total_monthly_savings = sum(r.monthly_savings for r in self.recommendations)
        total_upfront = sum(r.upfront_cost for r in self.recommendations)
        
        by_type = defaultdict(lambda: {'count': 0, 'savings': 0})
        for rec in self.recommendations:
            by_type[rec.recommendation_type.value]['count'] += 1
            by_type[rec.recommendation_type.value]['savings'] += rec.monthly_savings
        
        return {
            'current_reservations': {
                'total': total_current_ris,
                'underutilized': sum(1 for r in self.current_reservations if r.is_underutilized),
                'expiring_soon': sum(1 for r in self.current_reservations if r.is_expiring_soon),
                'total_monthly_cost': sum(r.monthly_cost for r in self.current_reservations)
            },
            'recommendations': {
                'total': total_recommendations,
                'by_type': dict(by_type),
                'total_monthly_savings': total_monthly_savings,
                'total_annual_savings': total_monthly_savings * 12,
                'required_upfront': total_upfront,
                'average_roi': (total_monthly_savings * 12 / total_upfront * 100 
                              if total_upfront > 0 else float('inf')),
                'top_recommendations': [
                    {
                        'type': rec.recommendation_type.value,
                        'instance_family': rec.instance_family,
                        'monthly_savings': rec.monthly_savings,
                        'term': rec.term_length,
                        'payment': rec.payment_option.value,
                        'confidence': rec.confidence_level
                    }
                    for rec in sorted(self.recommendations, 
                                    key=lambda x: x.monthly_savings, 
                                    reverse=True)[:5]
                ]
            }
        }