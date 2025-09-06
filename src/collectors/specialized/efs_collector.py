"""
AWS EFS (Elastic File System) specialized collector.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class EFSCollector:
    """Collector for AWS EFS optimization opportunities."""
    
    def __init__(self, session: boto3.Session = None):
        self.session = session or boto3.Session()
        self.efs_client = self.session.client('efs')
        self.cloudwatch = self.session.client('cloudwatch')
        self.pricing_client = self.session.client('pricing', region_name='us-east-1')
    
    def collect(self) -> Dict[str, Any]:
        """Collect EFS data and optimization opportunities."""
        try:
            file_systems = self._get_file_systems()
            metrics = self._collect_metrics(file_systems)
            analysis = self._analyze_efs_usage(file_systems, metrics)
            recommendations = self._generate_recommendations(analysis)
            
            return {
                'file_systems': file_systems,
                'metrics': metrics,
                'analysis': analysis,
                'recommendations': recommendations,
                'summary': self._generate_summary(analysis, recommendations)
            }
        except Exception as e:
            logger.error(f"Error collecting EFS data: {str(e)}")
            return {}
    
    def _get_file_systems(self) -> List[Dict[str, Any]]:
        """Get all EFS file systems."""
        file_systems = []
        
        try:
            paginator = self.efs_client.get_paginator('describe_file_systems')
            
            for page in paginator.paginate():
                for fs in page['FileSystems']:
                    # Get mount targets
                    mount_targets = self._get_mount_targets(fs['FileSystemId'])
                    
                    # Get lifecycle configuration
                    lifecycle = self._get_lifecycle_configuration(fs['FileSystemId'])
                    
                    # Get access points
                    access_points = self._get_access_points(fs['FileSystemId'])
                    
                    file_systems.append({
                        'file_system_id': fs['FileSystemId'],
                        'name': fs.get('Name', 'N/A'),
                        'creation_time': fs['CreationTime'],
                        'performance_mode': fs['PerformanceMode'],
                        'throughput_mode': fs.get('ThroughputMode', 'bursting'),
                        'provisioned_throughput': fs.get('ProvisionedThroughputInMibps'),
                        'size_bytes': fs['SizeInBytes']['Value'],
                        'lifecycle_state': fs['LifeCycleState'],
                        'encrypted': fs.get('Encrypted', False),
                        'mount_targets': mount_targets,
                        'lifecycle_policy': lifecycle,
                        'access_points': access_points,
                        'tags': fs.get('Tags', [])
                    })
        except ClientError as e:
            logger.error(f"Error getting file systems: {str(e)}")
        
        return file_systems
    
    def _get_mount_targets(self, file_system_id: str) -> List[Dict[str, Any]]:
        """Get mount targets for a file system."""
        mount_targets = []
        
        try:
            response = self.efs_client.describe_mount_targets(
                FileSystemId=file_system_id
            )
            
            for mt in response['MountTargets']:
                mount_targets.append({
                    'mount_target_id': mt['MountTargetId'],
                    'subnet_id': mt['SubnetId'],
                    'availability_zone': mt.get('AvailabilityZoneName'),
                    'lifecycle_state': mt['LifeCycleState']
                })
        except ClientError as e:
            logger.error(f"Error getting mount targets: {str(e)}")
        
        return mount_targets
    
    def _get_lifecycle_configuration(self, file_system_id: str) -> Optional[Dict[str, Any]]:
        """Get lifecycle configuration for a file system."""
        try:
            response = self.efs_client.describe_lifecycle_configuration(
                FileSystemId=file_system_id
            )
            
            return {
                'transition_to_ia': response.get('LifecyclePolicies', []),
                'has_lifecycle': len(response.get('LifecyclePolicies', [])) > 0
            }
        except ClientError:
            return None
    
    def _get_access_points(self, file_system_id: str) -> List[Dict[str, Any]]:
        """Get access points for a file system."""
        access_points = []
        
        try:
            response = self.efs_client.describe_access_points(
                FileSystemId=file_system_id
            )
            
            for ap in response['AccessPoints']:
                access_points.append({
                    'access_point_id': ap['AccessPointId'],
                    'lifecycle_state': ap['LifeCycleState'],
                    'root_directory': ap.get('RootDirectory', {}).get('Path', '/')
                })
        except ClientError as e:
            logger.error(f"Error getting access points: {str(e)}")
        
        return access_points
    
    def _collect_metrics(self, file_systems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect CloudWatch metrics for EFS file systems."""
        metrics = {}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        for fs in file_systems:
            fs_id = fs['file_system_id']
            metrics[fs_id] = {}
            
            # Collect various metrics
            metric_queries = [
                ('ClientConnections', 'Average', 'client_connections'),
                ('DataReadIOBytes', 'Sum', 'data_read_bytes'),
                ('DataWriteIOBytes', 'Sum', 'data_write_bytes'),
                ('MetadataIOBytes', 'Sum', 'metadata_bytes'),
                ('BurstCreditBalance', 'Average', 'burst_credits'),
                ('PercentIOLimit', 'Average', 'io_limit_percentage')
            ]
            
            for metric_name, stat, key in metric_queries:
                try:
                    response = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/EFS',
                        MetricName=metric_name,
                        Dimensions=[
                            {'Name': 'FileSystemId', 'Value': fs_id}
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=[stat]
                    )
                    
                    if response['Datapoints']:
                        values = [dp[stat] for dp in response['Datapoints']]
                        metrics[fs_id][key] = {
                            'average': sum(values) / len(values),
                            'max': max(values),
                            'min': min(values),
                            'latest': sorted(response['Datapoints'], 
                                          key=lambda x: x['Timestamp'])[-1][stat]
                        }
                except ClientError as e:
                    logger.error(f"Error getting metric {metric_name}: {str(e)}")
        
        return metrics
    
    def _analyze_efs_usage(self, file_systems: List[Dict[str, Any]], 
                          metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze EFS usage patterns."""
        analysis = {
            'underutilized': [],
            'no_lifecycle': [],
            'burst_credit_risk': [],
            'provisioned_throughput_waste': [],
            'single_az': [],
            'high_metadata_usage': []
        }
        
        for fs in file_systems:
            fs_id = fs['file_system_id']
            fs_metrics = metrics.get(fs_id, {})
            
            # Check for underutilization
            connections = fs_metrics.get('client_connections', {}).get('average', 0)
            if connections < 1:
                analysis['underutilized'].append({
                    'file_system_id': fs_id,
                    'name': fs['name'],
                    'size_gb': fs['size_bytes'] / (1024**3),
                    'avg_connections': connections
                })
            
            # Check lifecycle policy
            if not fs.get('lifecycle_policy', {}).get('has_lifecycle'):
                size_gb = fs['size_bytes'] / (1024**3)
                if size_gb > 10:  # Only flag if > 10GB
                    analysis['no_lifecycle'].append({
                        'file_system_id': fs_id,
                        'name': fs['name'],
                        'size_gb': size_gb,
                        'potential_savings': size_gb * 0.016  # IA is ~80% cheaper
                    })
            
            # Check burst credit balance
            burst_credits = fs_metrics.get('burst_credits', {}).get('latest', 0)
            if burst_credits < 1000000000:  # Less than 1TB of burst credits
                analysis['burst_credit_risk'].append({
                    'file_system_id': fs_id,
                    'name': fs['name'],
                    'burst_credits': burst_credits,
                    'throughput_mode': fs['throughput_mode']
                })
            
            # Check provisioned throughput efficiency
            if fs['throughput_mode'] == 'provisioned':
                io_limit = fs_metrics.get('io_limit_percentage', {}).get('average', 0)
                if io_limit < 50:
                    analysis['provisioned_throughput_waste'].append({
                        'file_system_id': fs_id,
                        'name': fs['name'],
                        'provisioned_mibps': fs['provisioned_throughput'],
                        'utilization_percent': io_limit
                    })
            
            # Check for single AZ deployment
            if len(fs['mount_targets']) == 1:
                analysis['single_az'].append({
                    'file_system_id': fs_id,
                    'name': fs['name'],
                    'availability_zone': fs['mount_targets'][0].get('availability_zone')
                })
            
            # Check metadata usage
            metadata_bytes = fs_metrics.get('metadata_bytes', {}).get('average', 0)
            total_bytes = (fs_metrics.get('data_read_bytes', {}).get('average', 0) +
                          fs_metrics.get('data_write_bytes', {}).get('average', 0))
            
            if total_bytes > 0:
                metadata_ratio = metadata_bytes / total_bytes
                if metadata_ratio > 0.3:  # More than 30% metadata
                    analysis['high_metadata_usage'].append({
                        'file_system_id': fs_id,
                        'name': fs['name'],
                        'metadata_ratio': metadata_ratio
                    })
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate EFS optimization recommendations."""
        recommendations = []
        
        # Underutilized file systems
        for fs in analysis['underutilized']:
            recommendations.append({
                'type': 'efs_underutilized',
                'resource_id': fs['file_system_id'],
                'title': f"Underutilized EFS: {fs['name']}",
                'description': f"EFS has average of {fs['avg_connections']:.1f} connections. Consider consolidating or removing.",
                'estimated_savings': fs['size_gb'] * 0.30,  # $0.30/GB/month for standard
                'priority': 'medium',
                'risk_level': 'low'
            })
        
        # Missing lifecycle policies
        for fs in analysis['no_lifecycle']:
            recommendations.append({
                'type': 'efs_no_lifecycle',
                'resource_id': fs['file_system_id'],
                'title': f"Enable Lifecycle Management: {fs['name']}",
                'description': f"Enable lifecycle to move infrequently accessed files to IA storage class",
                'estimated_savings': fs['potential_savings'],
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # Burst credit risk
        for fs in analysis['burst_credit_risk']:
            recommendations.append({
                'type': 'efs_burst_credit',
                'resource_id': fs['file_system_id'],
                'title': f"Low Burst Credits: {fs['name']}",
                'description': "Consider switching to provisioned throughput mode to avoid performance issues",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'medium'
            })
        
        # Provisioned throughput waste
        for fs in analysis['provisioned_throughput_waste']:
            recommendations.append({
                'type': 'efs_throughput_waste',
                'resource_id': fs['file_system_id'],
                'title': f"Overprovisioned Throughput: {fs['name']}",
                'description': f"Throughput utilization is only {fs['utilization_percent']:.1f}%. Consider reducing or switching to bursting mode.",
                'estimated_savings': fs['provisioned_mibps'] * 6.0 * 0.5,  # Assume 50% reduction possible
                'priority': 'medium',
                'risk_level': 'low'
            })
        
        # Single AZ deployment
        for fs in analysis['single_az']:
            recommendations.append({
                'type': 'efs_single_az',
                'resource_id': fs['file_system_id'],
                'title': f"Single AZ Risk: {fs['name']}",
                'description': "File system has only one mount target. Add mount targets in other AZs for high availability.",
                'estimated_savings': 0,
                'priority': 'medium',
                'risk_level': 'high'
            })
        
        return recommendations
    
    def _generate_summary(self, analysis: Dict[str, Any], 
                         recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate EFS optimization summary."""
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        
        return {
            'total_file_systems': sum(len(v) for v in analysis.values()),
            'optimization_opportunities': len(recommendations),
            'potential_monthly_savings': total_savings,
            'underutilized_count': len(analysis['underutilized']),
            'no_lifecycle_count': len(analysis['no_lifecycle']),
            'burst_credit_risk_count': len(analysis['burst_credit_risk']),
            'provisioned_waste_count': len(analysis['provisioned_throughput_waste']),
            'single_az_count': len(analysis['single_az'])
        }