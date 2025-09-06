"""
AWS S3 specialized collector.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Collector:
    """Collector for AWS S3 optimization opportunities."""
    
    def __init__(self, session: boto3.Session = None):
        self.session = session or boto3.Session()
        self.s3_client = self.session.client('s3')
        self.cloudwatch = self.session.client('cloudwatch')
        self.s3_analytics = self.session.client('s3')
    
    def collect(self) -> Dict[str, Any]:
        """Collect S3 data and optimization opportunities."""
        try:
            buckets = self._get_buckets()
            metrics = self._collect_metrics(buckets)
            analysis = self._analyze_s3_usage(buckets, metrics)
            recommendations = self._generate_recommendations(analysis)
            
            return {
                'buckets': buckets,
                'metrics': metrics,
                'analysis': analysis,
                'recommendations': recommendations,
                'summary': self._generate_summary(analysis, recommendations)
            }
        except Exception as e:
            logger.error(f"Error collecting S3 data: {str(e)}")
            return {}
    
    def _get_buckets(self) -> List[Dict[str, Any]]:
        """Get all S3 buckets with detailed information."""
        buckets = []
        
        try:
            response = self.s3_client.list_buckets()
            
            for bucket in response['Buckets']:
                bucket_name = bucket['Name']
                bucket_info = {
                    'name': bucket_name,
                    'creation_date': bucket['CreationDate'],
                    'region': self._get_bucket_region(bucket_name),
                    'size_bytes': 0,
                    'object_count': 0,
                    'storage_classes': {},
                    'lifecycle_rules': [],
                    'versioning': False,
                    'encryption': False,
                    'logging': False,
                    'public_access_block': False,
                    'intelligent_tiering': False,
                    'replication': False,
                    'analytics': [],
                    'tags': {}
                }
                
                try:
                    # Get bucket size and object count
                    size_info = self._get_bucket_size(bucket_name)
                    bucket_info.update(size_info)
                    
                    # Get lifecycle configuration
                    bucket_info['lifecycle_rules'] = self._get_lifecycle_rules(bucket_name)
                    
                    # Get versioning status
                    bucket_info['versioning'] = self._get_versioning_status(bucket_name)
                    
                    # Get encryption status
                    bucket_info['encryption'] = self._get_encryption_status(bucket_name)
                    
                    # Get logging status
                    bucket_info['logging'] = self._get_logging_status(bucket_name)
                    
                    # Get public access block
                    bucket_info['public_access_block'] = self._get_public_access_block(bucket_name)
                    
                    # Get intelligent tiering configuration
                    bucket_info['intelligent_tiering'] = self._get_intelligent_tiering(bucket_name)
                    
                    # Get replication configuration
                    bucket_info['replication'] = self._get_replication_status(bucket_name)
                    
                    # Get analytics configurations
                    bucket_info['analytics'] = self._get_analytics_configurations(bucket_name)
                    
                    # Get tags
                    bucket_info['tags'] = self._get_bucket_tags(bucket_name)
                    
                except ClientError as e:
                    logger.error(f"Error getting details for bucket {bucket_name}: {str(e)}")
                
                buckets.append(bucket_info)
                
        except ClientError as e:
            logger.error(f"Error listing buckets: {str(e)}")
        
        return buckets
    
    def _get_bucket_region(self, bucket_name: str) -> str:
        """Get bucket region."""
        try:
            response = self.s3_client.get_bucket_location(Bucket=bucket_name)
            region = response.get('LocationConstraint')
            return region if region else 'us-east-1'
        except ClientError:
            return 'unknown'
    
    def _get_bucket_size(self, bucket_name: str) -> Dict[str, Any]:
        """Get bucket size and object count by storage class."""
        size_info = {
            'size_bytes': 0,
            'object_count': 0,
            'storage_classes': {}
        }
        
        try:
            # Use CloudWatch metrics for bucket size
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)
            
            # Get total bucket size
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                size_info['size_bytes'] = response['Datapoints'][0]['Average']
            
            # Get object count
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='NumberOfObjects',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                size_info['object_count'] = int(response['Datapoints'][0]['Average'])
            
            # Get size by storage class
            storage_classes = ['STANDARD', 'STANDARD_IA', 'ONEZONE_IA', 'GLACIER', 'DEEP_ARCHIVE', 'INTELLIGENT_TIERING']
            
            for storage_class in storage_classes:
                try:
                    response = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/S3',
                        MetricName='BucketSizeBytes',
                        Dimensions=[
                            {'Name': 'BucketName', 'Value': bucket_name},
                            {'Name': 'StorageType', 'Value': f'{storage_class}Storage'}
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=['Average']
                    )
                    
                    if response['Datapoints']:
                        size_info['storage_classes'][storage_class] = response['Datapoints'][0]['Average']
                except ClientError:
                    pass
                    
        except ClientError as e:
            logger.error(f"Error getting bucket size for {bucket_name}: {str(e)}")
        
        return size_info
    
    def _get_lifecycle_rules(self, bucket_name: str) -> List[Dict[str, Any]]:
        """Get lifecycle rules for a bucket."""
        rules = []
        
        try:
            response = self.s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
            
            for rule in response.get('Rules', []):
                rules.append({
                    'id': rule.get('ID'),
                    'status': rule.get('Status'),
                    'transitions': rule.get('Transitions', []),
                    'expiration': rule.get('Expiration'),
                    'noncurrent_transitions': rule.get('NoncurrentVersionTransitions', [])
                })
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchLifecycleConfiguration':
                logger.error(f"Error getting lifecycle rules for {bucket_name}: {str(e)}")
        
        return rules
    
    def _get_versioning_status(self, bucket_name: str) -> bool:
        """Get versioning status for a bucket."""
        try:
            response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            return response.get('Status') == 'Enabled'
        except ClientError:
            return False
    
    def _get_encryption_status(self, bucket_name: str) -> bool:
        """Get encryption status for a bucket."""
        try:
            self.s3_client.get_bucket_encryption(Bucket=bucket_name)
            return True
        except ClientError:
            return False
    
    def _get_logging_status(self, bucket_name: str) -> bool:
        """Get logging status for a bucket."""
        try:
            response = self.s3_client.get_bucket_logging(Bucket=bucket_name)
            return 'LoggingEnabled' in response
        except ClientError:
            return False
    
    def _get_public_access_block(self, bucket_name: str) -> bool:
        """Get public access block status for a bucket."""
        try:
            response = self.s3_client.get_public_access_block(Bucket=bucket_name)
            config = response['PublicAccessBlockConfiguration']
            return (config.get('BlockPublicAcls', False) and 
                   config.get('IgnorePublicAcls', False) and 
                   config.get('BlockPublicPolicy', False) and 
                   config.get('RestrictPublicBuckets', False))
        except ClientError:
            return False
    
    def _get_intelligent_tiering(self, bucket_name: str) -> bool:
        """Check if intelligent tiering is configured."""
        try:
            response = self.s3_client.list_bucket_intelligent_tiering_configurations(
                Bucket=bucket_name
            )
            return len(response.get('IntelligentTieringConfigurationList', [])) > 0
        except ClientError:
            return False
    
    def _get_replication_status(self, bucket_name: str) -> bool:
        """Get replication status for a bucket."""
        try:
            self.s3_client.get_bucket_replication(Bucket=bucket_name)
            return True
        except ClientError:
            return False
    
    def _get_analytics_configurations(self, bucket_name: str) -> List[str]:
        """Get analytics configurations for a bucket."""
        configs = []
        
        try:
            response = self.s3_client.list_bucket_analytics_configurations(
                Bucket=bucket_name
            )
            
            for config in response.get('AnalyticsConfigurationList', []):
                configs.append(config.get('Id'))
        except ClientError:
            pass
        
        return configs
    
    def _get_bucket_tags(self, bucket_name: str) -> Dict[str, str]:
        """Get tags for a bucket."""
        try:
            response = self.s3_client.get_bucket_tagging(Bucket=bucket_name)
            return {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
        except ClientError:
            return {}
    
    def _collect_metrics(self, buckets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect CloudWatch metrics for S3 buckets."""
        metrics = {}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        
        for bucket in buckets:
            bucket_name = bucket['name']
            metrics[bucket_name] = {}
            
            # Collect request metrics
            metric_queries = [
                ('AllRequests', 'Sum', 'total_requests'),
                ('GetRequests', 'Sum', 'get_requests'),
                ('PutRequests', 'Sum', 'put_requests'),
                ('DeleteRequests', 'Sum', 'delete_requests'),
                ('ListRequests', 'Sum', 'list_requests'),
                ('4xxErrors', 'Sum', 'client_errors'),
                ('5xxErrors', 'Sum', 'server_errors')
            ]
            
            for metric_name, stat, key in metric_queries:
                try:
                    response = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/S3',
                        MetricName=metric_name,
                        Dimensions=[
                            {'Name': 'BucketName', 'Value': bucket_name}
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=[stat]
                    )
                    
                    if response['Datapoints']:
                        total = sum(dp[stat] for dp in response['Datapoints'])
                        metrics[bucket_name][key] = total
                except ClientError:
                    pass
        
        return metrics
    
    def _analyze_s3_usage(self, buckets: List[Dict[str, Any]], 
                         metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze S3 usage patterns."""
        analysis = {
            'unused_buckets': [],
            'no_lifecycle': [],
            'single_class_large': [],
            'unencrypted': [],
            'no_versioning': [],
            'public_access': [],
            'no_logging': [],
            'cross_region_replication': [],
            'incomplete_multipart': [],
            'old_versions': []
        }
        
        for bucket in buckets:
            bucket_name = bucket['name']
            bucket_metrics = metrics.get(bucket_name, {})
            
            # Check for unused buckets
            total_requests = bucket_metrics.get('total_requests', 0)
            age_days = (datetime.utcnow() - bucket['creation_date'].replace(tzinfo=None)).days
            
            if total_requests < 100 and age_days > 30:
                analysis['unused_buckets'].append({
                    'bucket_name': bucket_name,
                    'size_gb': bucket['size_bytes'] / (1024**3),
                    'object_count': bucket['object_count'],
                    'last_30d_requests': total_requests
                })
            
            # Check for missing lifecycle policies
            if not bucket['lifecycle_rules'] and bucket['size_bytes'] > 10 * (1024**3):  # > 10GB
                analysis['no_lifecycle'].append({
                    'bucket_name': bucket_name,
                    'size_gb': bucket['size_bytes'] / (1024**3),
                    'potential_savings': self._estimate_lifecycle_savings(bucket)
                })
            
            # Check for single storage class on large buckets
            if len(bucket['storage_classes']) <= 1 and bucket['size_bytes'] > 100 * (1024**3):  # > 100GB
                analysis['single_class_large'].append({
                    'bucket_name': bucket_name,
                    'size_gb': bucket['size_bytes'] / (1024**3),
                    'storage_class': list(bucket['storage_classes'].keys())[0] if bucket['storage_classes'] else 'STANDARD'
                })
            
            # Check encryption
            if not bucket['encryption']:
                analysis['unencrypted'].append({
                    'bucket_name': bucket_name,
                    'size_gb': bucket['size_bytes'] / (1024**3)
                })
            
            # Check versioning
            if not bucket['versioning'] and bucket['tags'].get('Environment') == 'Production':
                analysis['no_versioning'].append({
                    'bucket_name': bucket_name,
                    'environment': bucket['tags'].get('Environment', 'Unknown')
                })
            
            # Check public access
            if not bucket['public_access_block']:
                analysis['public_access'].append({
                    'bucket_name': bucket_name,
                    'region': bucket['region']
                })
            
            # Check logging
            if not bucket['logging'] and bucket['tags'].get('Environment') == 'Production':
                analysis['no_logging'].append({
                    'bucket_name': bucket_name
                })
            
            # Check cross-region replication costs
            if bucket['replication']:
                analysis['cross_region_replication'].append({
                    'bucket_name': bucket_name,
                    'size_gb': bucket['size_bytes'] / (1024**3)
                })
        
        return analysis
    
    def _estimate_lifecycle_savings(self, bucket: Dict[str, Any]) -> float:
        """Estimate savings from lifecycle policies."""
        size_gb = bucket['size_bytes'] / (1024**3)
        
        # Assume 30% could move to IA, 20% to Glacier
        ia_savings = size_gb * 0.3 * (0.023 - 0.0125)  # Standard to IA price difference
        glacier_savings = size_gb * 0.2 * (0.023 - 0.004)  # Standard to Glacier
        
        return ia_savings + glacier_savings
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate S3 optimization recommendations."""
        recommendations = []
        
        # Unused buckets
        for bucket in analysis['unused_buckets']:
            cost = bucket['size_gb'] * 0.023  # Standard storage cost
            
            recommendations.append({
                'type': 's3_unused',
                'resource_id': bucket['bucket_name'],
                'title': f"Remove unused bucket: {bucket['bucket_name']}",
                'description': f"Bucket has only {bucket['last_30d_requests']} requests in 30 days. Size: {bucket['size_gb']:.1f}GB",
                'estimated_savings': cost,
                'priority': 'medium',
                'risk_level': 'low'
            })
        
        # Missing lifecycle policies
        for bucket in analysis['no_lifecycle']:
            recommendations.append({
                'type': 's3_no_lifecycle',
                'resource_id': bucket['bucket_name'],
                'title': f"Enable lifecycle policy: {bucket['bucket_name']}",
                'description': f"Large bucket ({bucket['size_gb']:.1f}GB) without lifecycle rules. Enable tiering to save costs.",
                'estimated_savings': bucket['potential_savings'],
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # Single storage class
        for bucket in analysis['single_class_large']:
            recommendations.append({
                'type': 's3_single_class',
                'resource_id': bucket['bucket_name'],
                'title': f"Enable Intelligent-Tiering: {bucket['bucket_name']}",
                'description': f"Large bucket ({bucket['size_gb']:.1f}GB) using only {bucket['storage_class']}. Enable Intelligent-Tiering.",
                'estimated_savings': bucket['size_gb'] * 0.005,  # Estimate
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # Unencrypted buckets
        for bucket in analysis['unencrypted']:
            recommendations.append({
                'type': 's3_unencrypted',
                'resource_id': bucket['bucket_name'],
                'title': f"Enable encryption: {bucket['bucket_name']}",
                'description': "Bucket is not encrypted. Enable default encryption for security.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # No versioning
        for bucket in analysis['no_versioning']:
            recommendations.append({
                'type': 's3_no_versioning',
                'resource_id': bucket['bucket_name'],
                'title': f"Enable versioning: {bucket['bucket_name']}",
                'description': f"Production bucket without versioning. Enable for data protection.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # Public access
        for bucket in analysis['public_access']:
            recommendations.append({
                'type': 's3_public_access',
                'resource_id': bucket['bucket_name'],
                'title': f"Block public access: {bucket['bucket_name']}",
                'description': "Bucket doesn't have public access block enabled. Enable for security.",
                'estimated_savings': 0,
                'priority': 'critical',
                'risk_level': 'zero'
            })
        
        return recommendations
    
    def _generate_summary(self, analysis: Dict[str, Any], 
                         recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate S3 optimization summary."""
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        
        return {
            'total_buckets': sum(len(v) for v in analysis.values()),
            'optimization_opportunities': len(recommendations),
            'potential_monthly_savings': total_savings,
            'unused_buckets': len(analysis['unused_buckets']),
            'no_lifecycle_count': len(analysis['no_lifecycle']),
            'single_class_count': len(analysis['single_class_large']),
            'unencrypted_count': len(analysis['unencrypted']),
            'public_access_count': len(analysis['public_access'])
        }