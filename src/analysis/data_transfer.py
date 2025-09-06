"""
Data Transfer Cost Analyzer - Hidden AWS Cost Discovery
Analyzes data transfer costs across regions, AZs, and to internet.
Data transfer often represents 10-30% of AWS bills but is frequently overlooked.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TransferType(str, Enum):
    """Types of data transfer"""
    INTERNET_EGRESS = "internet_egress"
    INTER_REGION = "inter_region"
    INTER_AZ = "inter_az"
    CLOUDFRONT = "cloudfront"
    S3_TRANSFER = "s3_transfer"
    VPC_PEERING = "vpc_peering"
    DIRECT_CONNECT = "direct_connect"
    TRANSIT_GATEWAY = "transit_gateway"
    NAT_GATEWAY = "nat_gateway"
    VPC_ENDPOINT = "vpc_endpoint"


class OptimizationMethod(str, Enum):
    """Data transfer optimization methods"""
    USE_CLOUDFRONT = "use_cloudfront"
    VPC_ENDPOINT = "vpc_endpoint"
    SAME_AZ_PLACEMENT = "same_az_placement"
    DIRECT_CONNECT = "direct_connect"
    DATA_COMPRESSION = "data_compression"
    CACHING = "caching"
    BATCH_TRANSFERS = "batch_transfers"
    REGIONAL_SERVICES = "regional_services"
    S3_TRANSFER_ACCELERATION = "s3_transfer_acceleration"


@dataclass
class DataTransferFlow:
    """Represents a data transfer flow between two points"""
    flow_id: str
    source: str
    source_type: str  # ec2, s3, rds, etc.
    destination: str
    destination_type: str
    transfer_type: TransferType
    monthly_gb: float
    monthly_cost: float
    peak_daily_gb: float
    optimization_potential: float
    optimization_methods: List[OptimizationMethod]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def annual_cost(self) -> float:
        """Calculate annual cost"""
        return self.monthly_cost * 12
    
    @property
    def daily_average_gb(self) -> float:
        """Calculate daily average"""
        return self.monthly_gb / 30


@dataclass
class DataTransferRecommendation:
    """Data transfer optimization recommendation"""
    recommendation_id: str
    transfer_flow: DataTransferFlow
    current_monthly_cost: float
    optimized_monthly_cost: float
    monthly_savings: float
    savings_percentage: float
    optimization_method: OptimizationMethod
    implementation_steps: List[str]
    implementation_effort: str  # LOW, MEDIUM, HIGH
    confidence: str  # HIGH, MEDIUM, LOW
    risk_level: str  # ZERO, LOW, MEDIUM, HIGH
    prerequisites: List[str] = field(default_factory=list)
    
    @property
    def annual_savings(self) -> float:
        """Calculate annual savings"""
        return self.monthly_savings * 12


class DataTransferAnalyzer:
    """
    Analyzes data transfer costs and provides optimization recommendations.
    Focuses on the most expensive but often hidden data transfer charges.
    """
    
    # Data transfer pricing (USD per GB) - Approximate AWS pricing
    PRICING = {
        'internet_egress': {
            'first_1gb': 0.00,       # First 1GB free each month
            'next_9999gb': 0.09,     # Up to 10TB/month
            'next_40000gb': 0.085,   # 10-50TB/month  
            'next_100000gb': 0.070,  # 50-150TB/month
            'over_150000gb': 0.050,  # 150TB+/month
        },
        'inter_region': {
            'us_to_us': 0.02,        # US East to US West
            'us_to_eu': 0.02,        # US to Europe
            'us_to_ap': 0.08,        # US to Asia Pacific
            'us_to_sa': 0.16,        # US to South America
            'eu_to_eu': 0.02,        # Europe internal
            'ap_to_ap': 0.08,        # Asia Pacific internal
            'cross_continent': 0.12,  # Cross-continent transfers
            'to_china': 0.12,        # Special pricing to China
        },
        'inter_az': {
            'same_region': 0.01,     # Per GB between AZs in same region
        },
        'cloudfront': {
            'origin_fetch': 0.00,    # No charge for origin fetch
            'edge_to_internet': 0.085,  # Slightly less than direct egress
        },
        's3': {
            'cross_region_replication': 0.02,  # CRR pricing
            'transfer_acceleration': 0.04,      # Additional charge
            'to_internet': 0.09,                # Standard egress
        },
        'nat_gateway': {
            'processing': 0.045,      # Per GB processed
        },
        'vpc_endpoint': {
            'data_processing': 0.01,  # Per GB processed
        },
        'transit_gateway': {
            'data_processing': 0.02,  # Per GB processed
        }
    }
    
    # Optimization thresholds
    THRESHOLDS = {
        'high_transfer_gb': 1000,     # GB/month to consider high
        'high_cost_usd': 100,         # USD/month to prioritize
        'cross_az_threshold_gb': 500,  # GB/month to optimize AZ placement
        'cloudfront_threshold_gb': 5000,  # GB/month to consider CloudFront
        'compression_threshold_gb': 100,   # GB/month to suggest compression
    }
    
    def __init__(self, session=None):
        """
        Initialize Data Transfer Analyzer.
        
        Args:
            session: AWS session for API calls
        """
        self.session = session
        self.transfer_flows: List[DataTransferFlow] = []
        self.recommendations: List[DataTransferRecommendation] = []
        self._flow_counter = 0
        self._recommendation_counter = 0
    
    def analyze_vpc_flow_logs(self, flow_logs: List[Dict]) -> List[DataTransferFlow]:
        """
        Analyze VPC Flow Logs to identify data transfer patterns.
        
        Args:
            flow_logs: List of VPC flow log entries
            
        Returns:
            List of identified data transfer flows
        """
        flows_by_path = defaultdict(lambda: {
            'bytes': 0,
            'packets': 0,
            'count': 0,
            'sources': set(),
            'destinations': set()
        })
        
        for log in flow_logs:
            src_addr = log.get('srcaddr', 'unknown')
            dst_addr = log.get('dstaddr', 'unknown')
            bytes_transferred = log.get('bytes', 0)
            
            # Determine transfer type based on IP addresses
            transfer_type = self._determine_transfer_type(src_addr, dst_addr, log)
            
            path_key = f"{src_addr}→{dst_addr}:{transfer_type.value}"
            flows_by_path[path_key]['bytes'] += bytes_transferred
            flows_by_path[path_key]['packets'] += log.get('packets', 0)
            flows_by_path[path_key]['count'] += 1
            flows_by_path[path_key]['sources'].add(src_addr)
            flows_by_path[path_key]['destinations'].add(dst_addr)
        
        # Convert to DataTransferFlow objects
        for path, data in flows_by_path.items():
            monthly_gb = (data['bytes'] / (1024**3)) * 30  # Approximate monthly
            
            if monthly_gb < 1:  # Skip negligible transfers
                continue
            
            src, rest = path.split('→')
            dst, transfer_type_str = rest.split(':')
            transfer_type = TransferType(transfer_type_str)
            
            monthly_cost = self._calculate_transfer_cost(monthly_gb, transfer_type, src, dst)
            
            flow = DataTransferFlow(
                flow_id=self._generate_flow_id(),
                source=src,
                source_type=self._identify_resource_type(src),
                destination=dst,
                destination_type=self._identify_resource_type(dst),
                transfer_type=transfer_type,
                monthly_gb=monthly_gb,
                monthly_cost=monthly_cost,
                peak_daily_gb=monthly_gb / 30 * 1.5,  # Estimate peak
                optimization_potential=self._estimate_optimization_potential(transfer_type, monthly_gb),
                optimization_methods=self._suggest_optimization_methods(transfer_type, monthly_gb),
                metadata={
                    'packet_count': data['packets'],
                    'flow_count': data['count'],
                    'unique_sources': len(data['sources']),
                    'unique_destinations': len(data['destinations'])
                }
            )
            
            self.transfer_flows.append(flow)
        
        return self.transfer_flows
    
    def analyze_cloudwatch_metrics(self, metrics: Dict[str, Any]) -> List[DataTransferFlow]:
        """
        Analyze CloudWatch metrics for data transfer patterns.
        
        Args:
            metrics: CloudWatch metrics data
            
        Returns:
            List of identified data transfer flows
        """
        flows = []
        
        # Analyze NetworkIn/NetworkOut metrics
        for resource_id, resource_metrics in metrics.items():
            network_out_bytes = resource_metrics.get('NetworkOut', 0)
            network_in_bytes = resource_metrics.get('NetworkIn', 0)
            
            if network_out_bytes > 0:
                monthly_gb = (network_out_bytes / (1024**3)) * 30
                
                # Determine destination based on metrics
                if resource_metrics.get('is_public', False):
                    transfer_type = TransferType.INTERNET_EGRESS
                    destination = 'Internet'
                else:
                    transfer_type = TransferType.INTER_AZ
                    destination = 'Internal'
                
                monthly_cost = self._calculate_transfer_cost(
                    monthly_gb, transfer_type, resource_id, destination
                )
                
                flow = DataTransferFlow(
                    flow_id=self._generate_flow_id(),
                    source=resource_id,
                    source_type=resource_metrics.get('type', 'ec2'),
                    destination=destination,
                    destination_type='external' if destination == 'Internet' else 'internal',
                    transfer_type=transfer_type,
                    monthly_gb=monthly_gb,
                    monthly_cost=monthly_cost,
                    peak_daily_gb=monthly_gb / 30 * 1.5,
                    optimization_potential=self._estimate_optimization_potential(transfer_type, monthly_gb),
                    optimization_methods=self._suggest_optimization_methods(transfer_type, monthly_gb)
                )
                
                flows.append(flow)
                self.transfer_flows.append(flow)
        
        return flows
    
    def analyze_cost_and_usage_report(self, cur_data: List[Dict]) -> List[DataTransferFlow]:
        """
        Analyze AWS Cost and Usage Report for data transfer charges.
        
        Args:
            cur_data: Cost and Usage Report data
            
        Returns:
            List of identified data transfer flows
        """
        transfer_costs = defaultdict(lambda: {
            'cost': 0,
            'usage_amount': 0,
            'resources': set()
        })
        
        for item in cur_data:
            usage_type = item.get('lineItem/UsageType', '')
            
            # Filter for data transfer related usage types
            if any(keyword in usage_type.lower() for keyword in 
                   ['datatransfer', 'data-transfer', 'cloudfront', 'nat-gateway']):
                
                key = f"{item.get('product/ProductName', 'Unknown')}:{usage_type}"
                transfer_costs[key]['cost'] += float(item.get('lineItem/UnblendedCost', 0))
                transfer_costs[key]['usage_amount'] += float(item.get('lineItem/UsageAmount', 0))
                transfer_costs[key]['resources'].add(item.get('lineItem/ResourceId', 'unknown'))
        
        # Convert to DataTransferFlow objects
        flows = []
        for key, data in transfer_costs.items():
            if data['cost'] < 10:  # Skip small costs
                continue
            
            service, usage_type = key.split(':', 1)
            transfer_type = self._usage_type_to_transfer_type(usage_type)
            
            flow = DataTransferFlow(
                flow_id=self._generate_flow_id(),
                source=service,
                source_type='service',
                destination='Various',
                destination_type='mixed',
                transfer_type=transfer_type,
                monthly_gb=data['usage_amount'],
                monthly_cost=data['cost'],
                peak_daily_gb=data['usage_amount'] / 30 * 1.5,
                optimization_potential=self._estimate_optimization_potential(
                    transfer_type, data['usage_amount']
                ),
                optimization_methods=self._suggest_optimization_methods(
                    transfer_type, data['usage_amount']
                ),
                metadata={
                    'resource_count': len(data['resources']),
                    'usage_type': usage_type
                }
            )
            
            flows.append(flow)
            self.transfer_flows.append(flow)
        
        return flows
    
    def generate_recommendations(self) -> List[DataTransferRecommendation]:
        """
        Generate optimization recommendations for identified data transfer flows.
        
        Returns:
            List of data transfer optimization recommendations
        """
        self.recommendations = []
        
        for flow in self.transfer_flows:
            # Skip low-cost flows
            if flow.monthly_cost < self.THRESHOLDS['high_cost_usd'] / 10:
                continue
            
            # Generate recommendations based on transfer type
            if flow.transfer_type == TransferType.INTERNET_EGRESS:
                self._recommend_cloudfront_optimization(flow)
                self._recommend_compression(flow)
                
            elif flow.transfer_type == TransferType.INTER_REGION:
                self._recommend_regional_optimization(flow)
                self._recommend_data_replication(flow)
                
            elif flow.transfer_type == TransferType.INTER_AZ:
                self._recommend_same_az_placement(flow)
                self._recommend_vpc_endpoint(flow)
                
            elif flow.transfer_type == TransferType.NAT_GATEWAY:
                self._recommend_nat_instance(flow)
                self._recommend_vpc_endpoint(flow)
        
        # Sort recommendations by savings potential
        self.recommendations.sort(key=lambda x: x.monthly_savings, reverse=True)
        
        return self.recommendations
    
    def _recommend_cloudfront_optimization(self, flow: DataTransferFlow) -> None:
        """Generate CloudFront optimization recommendation"""
        if flow.monthly_gb < self.THRESHOLDS['cloudfront_threshold_gb']:
            return
        
        # Calculate CloudFront costs
        cf_distribution_cost = 0  # Free tier or existing distribution
        cf_request_cost = flow.monthly_gb * 0.0001  # Approximate request costs
        cf_data_cost = flow.monthly_gb * self.PRICING['cloudfront']['edge_to_internet']
        
        total_cf_cost = cf_distribution_cost + cf_request_cost + cf_data_cost
        
        if total_cf_cost < flow.monthly_cost:
            savings = flow.monthly_cost - total_cf_cost
            
            recommendation = DataTransferRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                transfer_flow=flow,
                current_monthly_cost=flow.monthly_cost,
                optimized_monthly_cost=total_cf_cost,
                monthly_savings=savings,
                savings_percentage=(savings / flow.monthly_cost) * 100,
                optimization_method=OptimizationMethod.USE_CLOUDFRONT,
                implementation_steps=[
                    "Create CloudFront distribution",
                    "Configure origin to point to current source",
                    "Update DNS to point to CloudFront",
                    "Configure caching policies",
                    "Monitor cache hit ratio"
                ],
                implementation_effort="MEDIUM",
                confidence="HIGH",
                risk_level="LOW",
                prerequisites=["DNS control", "HTTPS certificate"]
            )
            
            self.recommendations.append(recommendation)
    
    def _recommend_same_az_placement(self, flow: DataTransferFlow) -> None:
        """Recommend same-AZ placement for inter-AZ transfers"""
        if flow.monthly_gb < self.THRESHOLDS['cross_az_threshold_gb']:
            return
        
        # All inter-AZ transfer would be eliminated
        savings = flow.monthly_cost
        
        recommendation = DataTransferRecommendation(
            recommendation_id=self._generate_recommendation_id(),
            transfer_flow=flow,
            current_monthly_cost=flow.monthly_cost,
            optimized_monthly_cost=0,
            monthly_savings=savings,
            savings_percentage=100,
            optimization_method=OptimizationMethod.SAME_AZ_PLACEMENT,
            implementation_steps=[
                "Identify resources in different AZs",
                "Plan migration to same AZ",
                "Consider Multi-AZ requirements",
                "Migrate during maintenance window",
                "Update connection strings"
            ],
            implementation_effort="HIGH",
            confidence="HIGH",
            risk_level="MEDIUM",
            prerequisites=["Downtime window", "HA strategy review"]
        )
        
        self.recommendations.append(recommendation)
    
    def _recommend_vpc_endpoint(self, flow: DataTransferFlow) -> None:
        """Recommend VPC Endpoint for S3/DynamoDB transfers"""
        if flow.destination_type not in ['s3', 'dynamodb']:
            return
        
        # VPC Endpoint eliminates data transfer costs for S3/DynamoDB
        endpoint_cost = 0.01 * flow.monthly_gb  # Processing cost
        savings = flow.monthly_cost - endpoint_cost
        
        if savings > 50:  # Minimum $50/month savings
            recommendation = DataTransferRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                transfer_flow=flow,
                current_monthly_cost=flow.monthly_cost,
                optimized_monthly_cost=endpoint_cost,
                monthly_savings=savings,
                savings_percentage=(savings / flow.monthly_cost) * 100,
                optimization_method=OptimizationMethod.VPC_ENDPOINT,
                implementation_steps=[
                    f"Create VPC Endpoint for {flow.destination_type.upper()}",
                    "Update route tables",
                    "Configure endpoint policies",
                    "Test connectivity",
                    "Remove NAT Gateway routes if applicable"
                ],
                implementation_effort="LOW",
                confidence="HIGH",
                risk_level="LOW",
                prerequisites=["VPC configuration access"]
            )
            
            self.recommendations.append(recommendation)
    
    def _recommend_compression(self, flow: DataTransferFlow) -> None:
        """Recommend data compression"""
        if flow.monthly_gb < self.THRESHOLDS['compression_threshold_gb']:
            return
        
        # Estimate 50-70% compression for typical data
        compression_ratio = 0.4  # 60% reduction
        compressed_gb = flow.monthly_gb * compression_ratio
        compressed_cost = self._calculate_transfer_cost(
            compressed_gb, flow.transfer_type, flow.source, flow.destination
        )
        
        savings = flow.monthly_cost - compressed_cost
        
        if savings > 20:
            recommendation = DataTransferRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                transfer_flow=flow,
                current_monthly_cost=flow.monthly_cost,
                optimized_monthly_cost=compressed_cost,
                monthly_savings=savings,
                savings_percentage=(savings / flow.monthly_cost) * 100,
                optimization_method=OptimizationMethod.DATA_COMPRESSION,
                implementation_steps=[
                    "Implement gzip/brotli compression",
                    "Configure compression at application level",
                    "Test compression ratios",
                    "Monitor CPU impact",
                    "Adjust compression levels"
                ],
                implementation_effort="LOW",
                confidence="MEDIUM",
                risk_level="LOW",
                prerequisites=["Application code access"]
            )
            
            self.recommendations.append(recommendation)
    
    def _recommend_regional_optimization(self, flow: DataTransferFlow) -> None:
        """Recommend regional service deployment"""
        # For significant inter-region transfers, suggest regional deployment
        if flow.monthly_cost < 500:
            return
        
        # Assume 80% reduction by deploying regionally
        regional_cost = flow.monthly_cost * 0.2
        savings = flow.monthly_cost - regional_cost
        
        recommendation = DataTransferRecommendation(
            recommendation_id=self._generate_recommendation_id(),
            transfer_flow=flow,
            current_monthly_cost=flow.monthly_cost,
            optimized_monthly_cost=regional_cost,
            monthly_savings=savings,
            savings_percentage=80,
            optimization_method=OptimizationMethod.REGIONAL_SERVICES,
            implementation_steps=[
                "Deploy services in target region",
                "Set up data replication",
                "Configure geo-routing",
                "Test regional failover",
                "Monitor latency improvements"
            ],
            implementation_effort="HIGH",
            confidence="MEDIUM",
            risk_level="MEDIUM",
            prerequisites=["Multi-region architecture", "Data residency compliance"]
        )
        
        self.recommendations.append(recommendation)
    
    def _recommend_data_replication(self, flow: DataTransferFlow) -> None:
        """Recommend data replication strategies"""
        if flow.source_type != 's3' or flow.monthly_gb < 1000:
            return
        
        # S3 Cross-Region Replication vs on-demand transfer
        crr_cost = flow.monthly_gb * self.PRICING['s3']['cross_region_replication']
        
        if crr_cost < flow.monthly_cost:
            savings = flow.monthly_cost - crr_cost
            
            recommendation = DataTransferRecommendation(
                recommendation_id=self._generate_recommendation_id(),
                transfer_flow=flow,
                current_monthly_cost=flow.monthly_cost,
                optimized_monthly_cost=crr_cost,
                monthly_savings=savings,
                savings_percentage=(savings / flow.monthly_cost) * 100,
                optimization_method=OptimizationMethod.BATCH_TRANSFERS,
                implementation_steps=[
                    "Enable S3 Cross-Region Replication",
                    "Configure replication rules",
                    "Set up destination bucket",
                    "Monitor replication metrics",
                    "Implement lifecycle policies"
                ],
                implementation_effort="MEDIUM",
                confidence="HIGH",
                risk_level="LOW",
                prerequisites=["S3 versioning enabled", "Destination bucket setup"]
            )
            
            self.recommendations.append(recommendation)
    
    def _recommend_nat_instance(self, flow: DataTransferFlow) -> None:
        """Recommend NAT Instance for small workloads"""
        if flow.monthly_cost < 100:  # NAT Gateway base cost ~$45/month
            # For small workloads, NAT instance might be cheaper
            nat_instance_cost = 20  # t3.micro cost estimate
            nat_instance_transfer = flow.monthly_gb * 0.01  # Reduced processing cost
            total_cost = nat_instance_cost + nat_instance_transfer
            
            if total_cost < flow.monthly_cost:
                savings = flow.monthly_cost - total_cost
                
                recommendation = DataTransferRecommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    transfer_flow=flow,
                    current_monthly_cost=flow.monthly_cost,
                    optimized_monthly_cost=total_cost,
                    monthly_savings=savings,
                    savings_percentage=(savings / flow.monthly_cost) * 100,
                    optimization_method=OptimizationMethod.REGIONAL_SERVICES,
                    implementation_steps=[
                        "Launch NAT instance (t3.micro)",
                        "Configure source/destination check",
                        "Update route tables",
                        "Implement HA with scripts",
                        "Monitor instance health"
                    ],
                    implementation_effort="MEDIUM",
                    confidence="MEDIUM",
                    risk_level="MEDIUM",
                    prerequisites=["EC2 management skills", "HA scripting"]
                )
                
                self.recommendations.append(recommendation)
    
    # Helper methods
    def _determine_transfer_type(self, src: str, dst: str, log: Dict) -> TransferType:
        """Determine transfer type from IP addresses and metadata"""
        # Simplified logic - enhance based on actual IP ranges
        if self._is_public_ip(dst):
            return TransferType.INTERNET_EGRESS
        elif log.get('az') != log.get('dst_az'):
            return TransferType.INTER_AZ
        elif log.get('region') != log.get('dst_region'):
            return TransferType.INTER_REGION
        else:
            return TransferType.INTER_AZ
    
    def _is_public_ip(self, ip: str) -> bool:
        """Check if IP is public"""
        # Simplified check - implement proper IP range validation
        private_ranges = ['10.', '172.16.', '172.17.', '172.18.', '172.19.',
                         '172.20.', '172.21.', '172.22.', '172.23.', '172.24.',
                         '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
                         '172.30.', '172.31.', '192.168.']
        
        return not any(ip.startswith(range_prefix) for range_prefix in private_ranges)
    
    def _identify_resource_type(self, identifier: str) -> str:
        """Identify resource type from identifier"""
        if identifier.startswith('i-'):
            return 'ec2'
        elif identifier.startswith('arn:aws:s3'):
            return 's3'
        elif identifier.startswith('db-'):
            return 'rds'
        elif self._is_public_ip(identifier):
            return 'external'
        else:
            return 'unknown'
    
    def _calculate_transfer_cost(self, gb: float, transfer_type: TransferType,
                                 source: str, destination: str) -> float:
        """Calculate data transfer cost"""
        if transfer_type == TransferType.INTERNET_EGRESS:
            return self._calculate_tiered_egress_cost(gb)
        elif transfer_type == TransferType.INTER_REGION:
            return gb * self._get_inter_region_price(source, destination)
        elif transfer_type == TransferType.INTER_AZ:
            return gb * self.PRICING['inter_az']['same_region']
        elif transfer_type == TransferType.NAT_GATEWAY:
            return gb * self.PRICING['nat_gateway']['processing']
        else:
            return gb * 0.02  # Default pricing
    
    def _calculate_tiered_egress_cost(self, gb: float) -> float:
        """Calculate tiered internet egress cost"""
        cost = 0
        remaining = gb
        
        tiers = [
            (1, self.PRICING['internet_egress']['first_1gb']),
            (9999, self.PRICING['internet_egress']['next_9999gb']),
            (40000, self.PRICING['internet_egress']['next_40000gb']),
            (100000, self.PRICING['internet_egress']['next_100000gb']),
            (float('inf'), self.PRICING['internet_egress']['over_150000gb'])
        ]
        
        for tier_size, tier_price in tiers:
            if remaining <= 0:
                break
            tier_usage = min(remaining, tier_size)
            cost += tier_usage * tier_price
            remaining -= tier_usage
        
        return cost
    
    def _get_inter_region_price(self, source: str, destination: str) -> float:
        """Get inter-region transfer price"""
        # Simplified - implement actual region mapping
        if 'us-' in source and 'us-' in destination:
            return self.PRICING['inter_region']['us_to_us']
        elif 'us-' in source and 'eu-' in destination:
            return self.PRICING['inter_region']['us_to_eu']
        elif 'us-' in source and 'ap-' in destination:
            return self.PRICING['inter_region']['us_to_ap']
        else:
            return self.PRICING['inter_region']['cross_continent']
    
    def _estimate_optimization_potential(self, transfer_type: TransferType,
                                        monthly_gb: float) -> float:
        """Estimate optimization potential (0-100%)"""
        if transfer_type == TransferType.INTERNET_EGRESS and monthly_gb > 5000:
            return 40  # CloudFront can save ~40%
        elif transfer_type == TransferType.INTER_AZ:
            return 100  # Can eliminate completely
        elif transfer_type == TransferType.INTER_REGION:
            return 80  # Regional deployment can save ~80%
        elif transfer_type == TransferType.NAT_GATEWAY:
            return 50  # VPC Endpoints can save ~50%
        else:
            return 20  # Default potential
    
    def _suggest_optimization_methods(self, transfer_type: TransferType,
                                     monthly_gb: float) -> List[OptimizationMethod]:
        """Suggest applicable optimization methods"""
        methods = []
        
        if transfer_type == TransferType.INTERNET_EGRESS:
            methods.append(OptimizationMethod.USE_CLOUDFRONT)
            methods.append(OptimizationMethod.DATA_COMPRESSION)
            if monthly_gb > 10000:
                methods.append(OptimizationMethod.DIRECT_CONNECT)
        
        elif transfer_type == TransferType.INTER_AZ:
            methods.append(OptimizationMethod.SAME_AZ_PLACEMENT)
            methods.append(OptimizationMethod.CACHING)
        
        elif transfer_type == TransferType.INTER_REGION:
            methods.append(OptimizationMethod.REGIONAL_SERVICES)
            methods.append(OptimizationMethod.BATCH_TRANSFERS)
        
        elif transfer_type == TransferType.NAT_GATEWAY:
            methods.append(OptimizationMethod.VPC_ENDPOINT)
        
        # Universal methods
        if monthly_gb > 100:
            methods.append(OptimizationMethod.DATA_COMPRESSION)
            methods.append(OptimizationMethod.CACHING)
        
        return methods
    
    def _usage_type_to_transfer_type(self, usage_type: str) -> TransferType:
        """Convert CUR usage type to transfer type"""
        usage_lower = usage_type.lower()
        
        if 'datatransfer-out' in usage_lower:
            return TransferType.INTERNET_EGRESS
        elif 'datatransfer-regional' in usage_lower:
            return TransferType.INTER_REGION
        elif 'datatransfer-az' in usage_lower:
            return TransferType.INTER_AZ
        elif 'cloudfront' in usage_lower:
            return TransferType.CLOUDFRONT
        elif 'nat-gateway' in usage_lower:
            return TransferType.NAT_GATEWAY
        else:
            return TransferType.INTER_AZ
    
    def _generate_flow_id(self) -> str:
        """Generate unique flow ID"""
        self._flow_counter += 1
        return f"flow-{self._flow_counter:04d}"
    
    def _generate_recommendation_id(self) -> str:
        """Generate unique recommendation ID"""
        self._recommendation_counter += 1
        return f"rec-dt-{self._recommendation_counter:04d}"
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of data transfer analysis"""
        if not self.transfer_flows:
            return {
                'total_flows': 0,
                'total_monthly_cost': 0,
                'total_monthly_gb': 0,
                'recommendations': 0,
                'potential_savings': 0
            }
        
        total_cost = sum(flow.monthly_cost for flow in self.transfer_flows)
        total_gb = sum(flow.monthly_gb for flow in self.transfer_flows)
        total_savings = sum(rec.monthly_savings for rec in self.recommendations)
        
        by_type = defaultdict(lambda: {'count': 0, 'cost': 0, 'gb': 0})
        for flow in self.transfer_flows:
            by_type[flow.transfer_type.value]['count'] += 1
            by_type[flow.transfer_type.value]['cost'] += flow.monthly_cost
            by_type[flow.transfer_type.value]['gb'] += flow.monthly_gb
        
        return {
            'total_flows': len(self.transfer_flows),
            'total_monthly_cost': total_cost,
            'total_annual_cost': total_cost * 12,
            'total_monthly_gb': total_gb,
            'by_transfer_type': dict(by_type),
            'top_flows': [
                {
                    'source': flow.source,
                    'destination': flow.destination,
                    'type': flow.transfer_type.value,
                    'monthly_cost': flow.monthly_cost,
                    'monthly_gb': flow.monthly_gb
                }
                for flow in sorted(self.transfer_flows, 
                                 key=lambda x: x.monthly_cost, 
                                 reverse=True)[:10]
            ],
            'recommendations': {
                'count': len(self.recommendations),
                'total_monthly_savings': total_savings,
                'total_annual_savings': total_savings * 12,
                'savings_percentage': (total_savings / total_cost * 100) if total_cost > 0 else 0,
                'top_recommendations': [
                    {
                        'method': rec.optimization_method.value,
                        'monthly_savings': rec.monthly_savings,
                        'effort': rec.implementation_effort,
                        'risk': rec.risk_level
                    }
                    for rec in self.recommendations[:5]
                ]
            }
        }