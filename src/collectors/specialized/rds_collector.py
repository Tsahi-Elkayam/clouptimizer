"""
AWS RDS specialized collector.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class RDSCollector:
    """Collector for AWS RDS optimization opportunities."""
    
    def __init__(self, session: boto3.Session = None):
        self.session = session or boto3.Session()
        self.rds_client = self.session.client('rds')
        self.cloudwatch = self.session.client('cloudwatch')
        self.ce_client = self.session.client('ce')  # Cost Explorer
    
    def collect(self) -> Dict[str, Any]:
        """Collect RDS data and optimization opportunities."""
        try:
            instances = self._get_rds_instances()
            clusters = self._get_rds_clusters()
            snapshots = self._get_snapshots()
            metrics = self._collect_metrics(instances, clusters)
            analysis = self._analyze_rds_usage(instances, clusters, snapshots, metrics)
            recommendations = self._generate_recommendations(analysis)
            
            return {
                'instances': instances,
                'clusters': clusters,
                'snapshots': snapshots,
                'metrics': metrics,
                'analysis': analysis,
                'recommendations': recommendations,
                'summary': self._generate_summary(analysis, recommendations)
            }
        except Exception as e:
            logger.error(f"Error collecting RDS data: {str(e)}")
            return {}
    
    def _get_rds_instances(self) -> List[Dict[str, Any]]:
        """Get all RDS instances."""
        instances = []
        
        try:
            paginator = self.rds_client.get_paginator('describe_db_instances')
            
            for page in paginator.paginate():
                for db in page['DBInstances']:
                    instances.append({
                        'instance_id': db['DBInstanceIdentifier'],
                        'instance_class': db['DBInstanceClass'],
                        'engine': db['Engine'],
                        'engine_version': db['EngineVersion'],
                        'allocated_storage': db['AllocatedStorage'],
                        'storage_type': db.get('StorageType', 'standard'),
                        'iops': db.get('Iops'),
                        'multi_az': db['MultiAZ'],
                        'status': db['DBInstanceStatus'],
                        'creation_time': db.get('InstanceCreateTime'),
                        'backup_retention': db['BackupRetentionPeriod'],
                        'backup_window': db.get('PreferredBackupWindow'),
                        'maintenance_window': db.get('PreferredMaintenanceWindow'),
                        'auto_minor_upgrade': db.get('AutoMinorVersionUpgrade', False),
                        'storage_encrypted': db.get('StorageEncrypted', False),
                        'kms_key': db.get('KmsKeyId'),
                        'publicly_accessible': db.get('PubliclyAccessible', False),
                        'vpc_security_groups': [sg['VpcSecurityGroupId'] 
                                               for sg in db.get('VpcSecurityGroups', [])],
                        'tags': {tag['Key']: tag['Value'] 
                                for tag in db.get('TagList', [])}
                    })
        except ClientError as e:
            logger.error(f"Error getting RDS instances: {str(e)}")
        
        return instances
    
    def _get_rds_clusters(self) -> List[Dict[str, Any]]:
        """Get all RDS clusters (Aurora)."""
        clusters = []
        
        try:
            paginator = self.rds_client.get_paginator('describe_db_clusters')
            
            for page in paginator.paginate():
                for cluster in page['DBClusters']:
                    # Get cluster members
                    members = []
                    for member in cluster.get('DBClusterMembers', []):
                        members.append({
                            'instance_id': member['DBInstanceIdentifier'],
                            'is_writer': member['IsClusterWriter'],
                            'promotion_tier': member.get('PromotionTier', 0)
                        })
                    
                    clusters.append({
                        'cluster_id': cluster['DBClusterIdentifier'],
                        'engine': cluster['Engine'],
                        'engine_version': cluster['EngineVersion'],
                        'engine_mode': cluster.get('EngineMode', 'provisioned'),
                        'status': cluster['Status'],
                        'multi_az': cluster.get('MultiAZ', False),
                        'allocated_storage': cluster.get('AllocatedStorage'),
                        'backup_retention': cluster['BackupRetentionPeriod'],
                        'storage_encrypted': cluster.get('StorageEncrypted', False),
                        'serverless_scaling': cluster.get('ScalingConfigurationInfo'),
                        'members': members,
                        'tags': {tag['Key']: tag['Value'] 
                                for tag in cluster.get('TagList', [])}
                    })
        except ClientError as e:
            logger.error(f"Error getting RDS clusters: {str(e)}")
        
        return clusters
    
    def _get_snapshots(self) -> List[Dict[str, Any]]:
        """Get RDS snapshots."""
        snapshots = []
        
        try:
            # Get manual snapshots
            paginator = self.rds_client.get_paginator('describe_db_snapshots')
            
            for page in paginator.paginate(SnapshotType='manual'):
                for snapshot in page['DBSnapshots']:
                    age_days = (datetime.utcnow() - snapshot['SnapshotCreateTime'].replace(tzinfo=None)).days
                    
                    snapshots.append({
                        'snapshot_id': snapshot['DBSnapshotIdentifier'],
                        'instance_id': snapshot['DBInstanceIdentifier'],
                        'engine': snapshot['Engine'],
                        'allocated_storage': snapshot['AllocatedStorage'],
                        'status': snapshot['Status'],
                        'creation_time': snapshot['SnapshotCreateTime'],
                        'age_days': age_days,
                        'encrypted': snapshot.get('Encrypted', False),
                        'type': 'manual'
                    })
            
            # Get automated snapshots
            for page in paginator.paginate(SnapshotType='automated'):
                for snapshot in page['DBSnapshots']:
                    age_days = (datetime.utcnow() - snapshot['SnapshotCreateTime'].replace(tzinfo=None)).days
                    
                    snapshots.append({
                        'snapshot_id': snapshot['DBSnapshotIdentifier'],
                        'instance_id': snapshot['DBInstanceIdentifier'],
                        'engine': snapshot['Engine'],
                        'allocated_storage': snapshot['AllocatedStorage'],
                        'status': snapshot['Status'],
                        'creation_time': snapshot['SnapshotCreateTime'],
                        'age_days': age_days,
                        'encrypted': snapshot.get('Encrypted', False),
                        'type': 'automated'
                    })
                    
        except ClientError as e:
            logger.error(f"Error getting snapshots: {str(e)}")
        
        return snapshots
    
    def _collect_metrics(self, instances: List[Dict[str, Any]], 
                        clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect CloudWatch metrics for RDS resources."""
        metrics = {}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        # Collect instance metrics
        for instance in instances:
            instance_id = instance['instance_id']
            metrics[instance_id] = self._get_instance_metrics(instance_id, start_time, end_time)
        
        # Collect cluster metrics
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            metrics[f"cluster_{cluster_id}"] = self._get_cluster_metrics(cluster_id, start_time, end_time)
        
        return metrics
    
    def _get_instance_metrics(self, instance_id: str, start_time: datetime, 
                             end_time: datetime) -> Dict[str, Any]:
        """Get metrics for a single RDS instance."""
        metrics = {}
        
        metric_queries = [
            ('CPUUtilization', 'Average', 'cpu_utilization'),
            ('DatabaseConnections', 'Average', 'connections'),
            ('FreeableMemory', 'Average', 'freeable_memory'),
            ('FreeStorageSpace', 'Average', 'free_storage'),
            ('ReadIOPS', 'Average', 'read_iops'),
            ('WriteIOPS', 'Average', 'write_iops'),
            ('ReadLatency', 'Average', 'read_latency'),
            ('WriteLatency', 'Average', 'write_latency'),
            ('NetworkReceiveThroughput', 'Average', 'network_rx'),
            ('NetworkTransmitThroughput', 'Average', 'network_tx')
        ]
        
        for metric_name, stat, key in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/RDS',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'DBInstanceIdentifier', 'Value': instance_id}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=[stat]
                )
                
                if response['Datapoints']:
                    values = [dp[stat] for dp in response['Datapoints']]
                    metrics[key] = {
                        'average': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values),
                        'latest': sorted(response['Datapoints'], 
                                       key=lambda x: x['Timestamp'])[-1][stat]
                    }
            except ClientError as e:
                logger.error(f"Error getting metric {metric_name} for {instance_id}: {str(e)}")
        
        return metrics
    
    def _get_cluster_metrics(self, cluster_id: str, start_time: datetime, 
                           end_time: datetime) -> Dict[str, Any]:
        """Get metrics for a single RDS cluster."""
        metrics = {}
        
        metric_queries = [
            ('CPUUtilization', 'Average', 'cpu_utilization'),
            ('DatabaseConnections', 'Average', 'connections'),
            ('VolumeBytesUsed', 'Average', 'volume_bytes_used'),
            ('VolumeReadIOPs', 'Average', 'read_iops'),
            ('VolumeWriteIOPs', 'Average', 'write_iops')
        ]
        
        for metric_name, stat, key in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/RDS',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'DBClusterIdentifier', 'Value': cluster_id}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=[stat]
                )
                
                if response['Datapoints']:
                    values = [dp[stat] for dp in response['Datapoints']]
                    metrics[key] = {
                        'average': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    }
            except ClientError as e:
                logger.error(f"Error getting metric {metric_name} for cluster {cluster_id}: {str(e)}")
        
        return metrics
    
    def _analyze_rds_usage(self, instances: List[Dict[str, Any]], 
                          clusters: List[Dict[str, Any]],
                          snapshots: List[Dict[str, Any]],
                          metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RDS usage patterns."""
        analysis = {
            'oversized_instances': [],
            'idle_instances': [],
            'old_snapshots': [],
            'unencrypted_resources': [],
            'single_az_production': [],
            'old_engine_versions': [],
            'high_iops_standard': [],
            'aurora_serverless_candidates': [],
            'backup_optimization': []
        }
        
        # Analyze instances
        for instance in instances:
            instance_id = instance['instance_id']
            instance_metrics = metrics.get(instance_id, {})
            
            # Check for oversized instances
            cpu_avg = instance_metrics.get('cpu_utilization', {}).get('average', 0)
            memory_free = instance_metrics.get('freeable_memory', {}).get('average', 0)
            
            if cpu_avg < 20:  # Less than 20% CPU utilization
                analysis['oversized_instances'].append({
                    'instance_id': instance_id,
                    'instance_class': instance['instance_class'],
                    'cpu_utilization': cpu_avg,
                    'engine': instance['engine']
                })
            
            # Check for idle instances
            connections = instance_metrics.get('connections', {}).get('average', 0)
            if connections < 1:
                analysis['idle_instances'].append({
                    'instance_id': instance_id,
                    'instance_class': instance['instance_class'],
                    'avg_connections': connections
                })
            
            # Check encryption
            if not instance['storage_encrypted']:
                analysis['unencrypted_resources'].append({
                    'resource_id': instance_id,
                    'resource_type': 'instance',
                    'engine': instance['engine']
                })
            
            # Check Multi-AZ for production
            if 'prod' in instance_id.lower() and not instance['multi_az']:
                analysis['single_az_production'].append({
                    'instance_id': instance_id,
                    'engine': instance['engine']
                })
            
            # Check for old engine versions
            if self._is_old_engine_version(instance['engine'], instance['engine_version']):
                analysis['old_engine_versions'].append({
                    'instance_id': instance_id,
                    'engine': instance['engine'],
                    'current_version': instance['engine_version']
                })
            
            # Check for high IOPS on standard storage
            if instance['storage_type'] == 'standard':
                read_iops = instance_metrics.get('read_iops', {}).get('average', 0)
                write_iops = instance_metrics.get('write_iops', {}).get('average', 0)
                total_iops = read_iops + write_iops
                
                if total_iops > 100:
                    analysis['high_iops_standard'].append({
                        'instance_id': instance_id,
                        'total_iops': total_iops,
                        'storage_type': instance['storage_type']
                    })
        
        # Analyze clusters
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            cluster_metrics = metrics.get(f"cluster_{cluster_id}", {})
            
            # Check for Aurora Serverless candidates
            if cluster['engine_mode'] == 'provisioned':
                cpu_avg = cluster_metrics.get('cpu_utilization', {}).get('average', 0)
                connections = cluster_metrics.get('connections', {}).get('average', 0)
                
                if cpu_avg < 30 and connections < 50:
                    analysis['aurora_serverless_candidates'].append({
                        'cluster_id': cluster_id,
                        'cpu_utilization': cpu_avg,
                        'avg_connections': connections,
                        'member_count': len(cluster['members'])
                    })
        
        # Analyze snapshots
        for snapshot in snapshots:
            if snapshot['type'] == 'manual' and snapshot['age_days'] > 90:
                analysis['old_snapshots'].append({
                    'snapshot_id': snapshot['snapshot_id'],
                    'instance_id': snapshot['instance_id'],
                    'age_days': snapshot['age_days'],
                    'storage_gb': snapshot['allocated_storage']
                })
        
        # Analyze backup optimization
        for instance in instances:
            if instance['backup_retention'] > 7:
                analysis['backup_optimization'].append({
                    'instance_id': instance['instance_id'],
                    'retention_days': instance['backup_retention'],
                    'suggested_retention': 7
                })
        
        return analysis
    
    def _is_old_engine_version(self, engine: str, version: str) -> bool:
        """Check if engine version is old."""
        # Simplified version check - in production, maintain a version database
        old_versions = {
            'mysql': ['5.6', '5.7'],
            'postgres': ['9.', '10.', '11.'],
            'mariadb': ['10.0', '10.1', '10.2']
        }
        
        for old_ver in old_versions.get(engine, []):
            if version.startswith(old_ver):
                return True
        
        return False
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate RDS optimization recommendations."""
        recommendations = []
        
        # Oversized instances
        for instance in analysis['oversized_instances']:
            current_class = instance['instance_class']
            suggested_class = self._suggest_smaller_instance(current_class)
            savings = self._estimate_instance_savings(current_class, suggested_class)
            
            recommendations.append({
                'type': 'rds_rightsizing',
                'resource_id': instance['instance_id'],
                'title': f"Rightsize RDS instance: {instance['instance_id']}",
                'description': f"CPU utilization is only {instance['cpu_utilization']:.1f}%. Consider downsizing from {current_class} to {suggested_class}",
                'estimated_savings': savings,
                'priority': 'high',
                'risk_level': 'medium'
            })
        
        # Idle instances
        for instance in analysis['idle_instances']:
            recommendations.append({
                'type': 'rds_idle',
                'resource_id': instance['instance_id'],
                'title': f"Idle RDS instance: {instance['instance_id']}",
                'description': f"Instance has {instance['avg_connections']:.1f} average connections. Consider removing or stopping.",
                'estimated_savings': self._estimate_instance_cost(instance['instance_class']),
                'priority': 'high',
                'risk_level': 'low'
            })
        
        # Old snapshots
        for snapshot in analysis['old_snapshots']:
            storage_cost = snapshot['storage_gb'] * 0.095  # $0.095/GB/month
            
            recommendations.append({
                'type': 'rds_old_snapshot',
                'resource_id': snapshot['snapshot_id'],
                'title': f"Old RDS snapshot: {snapshot['snapshot_id']}",
                'description': f"Snapshot is {snapshot['age_days']} days old. Consider deleting or moving to cheaper storage.",
                'estimated_savings': storage_cost,
                'priority': 'low',
                'risk_level': 'low'
            })
        
        # Unencrypted resources
        for resource in analysis['unencrypted_resources']:
            recommendations.append({
                'type': 'rds_unencrypted',
                'resource_id': resource['resource_id'],
                'title': f"Enable encryption: {resource['resource_id']}",
                'description': "Database is not encrypted. Enable encryption for security compliance.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # Single AZ production
        for instance in analysis['single_az_production']:
            recommendations.append({
                'type': 'rds_single_az',
                'resource_id': instance['instance_id'],
                'title': f"Enable Multi-AZ: {instance['instance_id']}",
                'description': "Production database is single-AZ. Enable Multi-AZ for high availability.",
                'estimated_savings': 0,
                'priority': 'critical',
                'risk_level': 'high'
            })
        
        # Aurora Serverless candidates
        for cluster in analysis['aurora_serverless_candidates']:
            recommendations.append({
                'type': 'rds_serverless',
                'resource_id': cluster['cluster_id'],
                'title': f"Consider Aurora Serverless: {cluster['cluster_id']}",
                'description': f"Low utilization ({cluster['cpu_utilization']:.1f}% CPU). Aurora Serverless could reduce costs.",
                'estimated_savings': 500,  # Estimate
                'priority': 'medium',
                'risk_level': 'medium'
            })
        
        # High IOPS on standard storage
        for instance in analysis['high_iops_standard']:
            recommendations.append({
                'type': 'rds_storage_upgrade',
                'resource_id': instance['instance_id'],
                'title': f"Upgrade storage type: {instance['instance_id']}",
                'description': f"High IOPS ({instance['total_iops']:.0f}) on standard storage. Upgrade to gp2 or io1.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'low'
            })
        
        return recommendations
    
    def _suggest_smaller_instance(self, current_class: str) -> str:
        """Suggest a smaller instance class."""
        # Simplified mapping - in production, use a comprehensive instance family tree
        downsizing_map = {
            'db.m5.24xlarge': 'db.m5.16xlarge',
            'db.m5.16xlarge': 'db.m5.12xlarge',
            'db.m5.12xlarge': 'db.m5.8xlarge',
            'db.m5.8xlarge': 'db.m5.4xlarge',
            'db.m5.4xlarge': 'db.m5.2xlarge',
            'db.m5.2xlarge': 'db.m5.xlarge',
            'db.m5.xlarge': 'db.m5.large',
            'db.m5.large': 'db.t3.medium',
            'db.r5.24xlarge': 'db.r5.16xlarge',
            'db.r5.16xlarge': 'db.r5.12xlarge',
            'db.r5.12xlarge': 'db.r5.8xlarge',
            'db.r5.8xlarge': 'db.r5.4xlarge',
            'db.r5.4xlarge': 'db.r5.2xlarge',
            'db.r5.2xlarge': 'db.r5.xlarge',
            'db.r5.xlarge': 'db.r5.large',
            'db.r5.large': 'db.t3.large'
        }
        
        return downsizing_map.get(current_class, 'db.t3.medium')
    
    def _estimate_instance_savings(self, current_class: str, suggested_class: str) -> float:
        """Estimate savings from instance resizing."""
        # Simplified pricing - in production, use AWS Pricing API
        monthly_costs = {
            'db.m5.24xlarge': 6650,
            'db.m5.16xlarge': 4430,
            'db.m5.12xlarge': 3320,
            'db.m5.8xlarge': 2215,
            'db.m5.4xlarge': 1110,
            'db.m5.2xlarge': 555,
            'db.m5.xlarge': 277,
            'db.m5.large': 139,
            'db.t3.medium': 50,
            'db.t3.large': 100
        }
        
        current_cost = monthly_costs.get(current_class, 500)
        suggested_cost = monthly_costs.get(suggested_class, 250)
        
        return max(0, current_cost - suggested_cost)
    
    def _estimate_instance_cost(self, instance_class: str) -> float:
        """Estimate monthly cost of an instance."""
        monthly_costs = {
            'db.m5.24xlarge': 6650,
            'db.m5.16xlarge': 4430,
            'db.m5.12xlarge': 3320,
            'db.m5.8xlarge': 2215,
            'db.m5.4xlarge': 1110,
            'db.m5.2xlarge': 555,
            'db.m5.xlarge': 277,
            'db.m5.large': 139,
            'db.t3.medium': 50,
            'db.t3.large': 100
        }
        
        return monthly_costs.get(instance_class, 200)
    
    def _generate_summary(self, analysis: Dict[str, Any], 
                         recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate RDS optimization summary."""
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        
        return {
            'total_instances': sum(len(v) for k, v in analysis.items() if 'instances' in k),
            'optimization_opportunities': len(recommendations),
            'potential_monthly_savings': total_savings,
            'oversized_count': len(analysis['oversized_instances']),
            'idle_count': len(analysis['idle_instances']),
            'old_snapshots_count': len(analysis['old_snapshots']),
            'unencrypted_count': len(analysis['unencrypted_resources']),
            'single_az_prod_count': len(analysis['single_az_production']),
            'serverless_candidates': len(analysis['aurora_serverless_candidates'])
        }