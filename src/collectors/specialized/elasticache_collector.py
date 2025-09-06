"""
AWS ElastiCache specialized collector.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ElastiCacheCollector:
    """Collector for AWS ElastiCache optimization opportunities."""
    
    def __init__(self, session: boto3.Session = None):
        self.session = session or boto3.Session()
        self.elasticache_client = self.session.client('elasticache')
        self.cloudwatch = self.session.client('cloudwatch')
    
    def collect(self) -> Dict[str, Any]:
        """Collect ElastiCache data and optimization opportunities."""
        try:
            clusters = self._get_cache_clusters()
            replication_groups = self._get_replication_groups()
            snapshots = self._get_snapshots()
            metrics = self._collect_metrics(clusters, replication_groups)
            analysis = self._analyze_elasticache_usage(clusters, replication_groups, snapshots, metrics)
            recommendations = self._generate_recommendations(analysis)
            
            return {
                'clusters': clusters,
                'replication_groups': replication_groups,
                'snapshots': snapshots,
                'metrics': metrics,
                'analysis': analysis,
                'recommendations': recommendations,
                'summary': self._generate_summary(analysis, recommendations)
            }
        except Exception as e:
            logger.error(f"Error collecting ElastiCache data: {str(e)}")
            return {}
    
    def _get_cache_clusters(self) -> List[Dict[str, Any]]:
        """Get all ElastiCache clusters."""
        clusters = []
        
        try:
            paginator = self.elasticache_client.get_paginator('describe_cache_clusters')
            
            for page in paginator.paginate(ShowCacheNodeInfo=True):
                for cluster in page['CacheClusters']:
                    cluster_info = {
                        'cluster_id': cluster['CacheClusterId'],
                        'engine': cluster['Engine'],
                        'engine_version': cluster['EngineVersion'],
                        'cache_node_type': cluster['CacheNodeType'],
                        'num_cache_nodes': cluster['NumCacheNodes'],
                        'cluster_status': cluster['CacheClusterStatus'],
                        'creation_time': cluster.get('CacheClusterCreateTime'),
                        'preferred_az': cluster.get('PreferredAvailabilityZone'),
                        'preferred_maintenance_window': cluster.get('PreferredMaintenanceWindow'),
                        'notification_arn': cluster.get('NotificationConfiguration', {}).get('TopicArn'),
                        'security_groups': [],
                        'parameter_group': cluster.get('CacheParameterGroup', {}).get('CacheParameterGroupName'),
                        'subnet_group': cluster.get('CacheSubnetGroupName'),
                        'replication_group_id': cluster.get('ReplicationGroupId'),
                        'snapshot_retention': cluster.get('SnapshotRetentionLimit', 0),
                        'auto_minor_upgrade': cluster.get('AutoMinorVersionUpgrade', False),
                        'tags': {}
                    }
                    
                    # Get cache nodes info
                    if 'CacheNodes' in cluster:
                        cluster_info['cache_nodes'] = []
                        for node in cluster['CacheNodes']:
                            cluster_info['cache_nodes'].append({
                                'node_id': node['CacheNodeId'],
                                'status': node['CacheNodeStatus'],
                                'creation_time': node.get('CacheNodeCreateTime'),
                                'endpoint': node.get('Endpoint', {}).get('Address')
                            })
                    
                    # Get security groups
                    for sg in cluster.get('CacheSecurityGroups', []):
                        cluster_info['security_groups'].append(sg['CacheSecurityGroupName'])
                    
                    # Get tags
                    try:
                        arn = f"arn:aws:elasticache:{self.session.region_name}:{cluster.get('OwnerId')}:cluster:{cluster['CacheClusterId']}"
                        tags_response = self.elasticache_client.list_tags_for_resource(ResourceName=arn)
                        cluster_info['tags'] = {tag['Key']: tag['Value'] for tag in tags_response.get('TagList', [])}
                    except ClientError:
                        pass
                    
                    clusters.append(cluster_info)
                    
        except ClientError as e:
            logger.error(f"Error getting cache clusters: {str(e)}")
        
        return clusters
    
    def _get_replication_groups(self) -> List[Dict[str, Any]]:
        """Get all ElastiCache replication groups."""
        replication_groups = []
        
        try:
            paginator = self.elasticache_client.get_paginator('describe_replication_groups')
            
            for page in paginator.paginate():
                for group in page['ReplicationGroups']:
                    group_info = {
                        'replication_group_id': group['ReplicationGroupId'],
                        'description': group.get('Description'),
                        'status': group['Status'],
                        'multi_az': group.get('MultiAZ', 'disabled'),
                        'automatic_failover': group.get('AutomaticFailover', 'disabled'),
                        'cache_node_type': group.get('CacheNodeType'),
                        'snapshot_retention': group.get('SnapshotRetentionLimit', 0),
                        'snapshot_window': group.get('SnapshotWindow'),
                        'cluster_mode': group.get('ClusterEnabled', False),
                        'auth_token_enabled': group.get('AuthTokenEnabled', False),
                        'transit_encryption': group.get('TransitEncryptionEnabled', False),
                        'at_rest_encryption': group.get('AtRestEncryptionEnabled', False),
                        'member_clusters': group.get('MemberClusters', []),
                        'node_groups': []
                    }
                    
                    # Get node groups info
                    for ng in group.get('NodeGroups', []):
                        node_group = {
                            'node_group_id': ng['NodeGroupId'],
                            'status': ng['Status'],
                            'primary_endpoint': ng.get('PrimaryEndpoint', {}).get('Address'),
                            'reader_endpoint': ng.get('ReaderEndpoint', {}).get('Address'),
                            'slots': ng.get('Slots'),
                            'node_group_members': []
                        }
                        
                        for member in ng.get('NodeGroupMembers', []):
                            node_group['node_group_members'].append({
                                'cache_cluster_id': member['CacheClusterId'],
                                'cache_node_id': member['CacheNodeId'],
                                'role': member.get('CurrentRole'),
                                'preferred_az': member.get('PreferredAvailabilityZone')
                            })
                        
                        group_info['node_groups'].append(node_group)
                    
                    replication_groups.append(group_info)
                    
        except ClientError as e:
            logger.error(f"Error getting replication groups: {str(e)}")
        
        return replication_groups
    
    def _get_snapshots(self) -> List[Dict[str, Any]]:
        """Get ElastiCache snapshots."""
        snapshots = []
        
        try:
            paginator = self.elasticache_client.get_paginator('describe_snapshots')
            
            for page in paginator.paginate():
                for snapshot in page['Snapshots']:
                    creation_time = snapshot.get('SnapshotCreateTime')
                    if creation_time:
                        age_days = (datetime.utcnow() - creation_time.replace(tzinfo=None)).days
                    else:
                        age_days = 0
                    
                    snapshots.append({
                        'snapshot_name': snapshot['SnapshotName'],
                        'cluster_id': snapshot.get('CacheClusterId'),
                        'replication_group_id': snapshot.get('ReplicationGroupId'),
                        'snapshot_status': snapshot['SnapshotStatus'],
                        'snapshot_source': snapshot.get('SnapshotSource'),
                        'engine': snapshot.get('Engine'),
                        'engine_version': snapshot.get('EngineVersion'),
                        'cache_node_type': snapshot.get('CacheNodeType'),
                        'creation_time': creation_time,
                        'age_days': age_days
                    })
                    
        except ClientError as e:
            logger.error(f"Error getting snapshots: {str(e)}")
        
        return snapshots
    
    def _collect_metrics(self, clusters: List[Dict[str, Any]], 
                        replication_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect CloudWatch metrics for ElastiCache resources."""
        metrics = {}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        # Collect cluster metrics
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            engine = cluster['engine']
            
            if engine == 'redis':
                metrics[cluster_id] = self._get_redis_metrics(cluster_id, start_time, end_time)
            elif engine == 'memcached':
                metrics[cluster_id] = self._get_memcached_metrics(cluster_id, start_time, end_time)
        
        # Collect replication group metrics
        for group in replication_groups:
            group_id = group['replication_group_id']
            metrics[f"rg_{group_id}"] = self._get_replication_group_metrics(group_id, start_time, end_time)
        
        return metrics
    
    def _get_redis_metrics(self, cluster_id: str, start_time: datetime, 
                          end_time: datetime) -> Dict[str, Any]:
        """Get Redis-specific metrics."""
        metrics = {}
        
        metric_queries = [
            ('CPUUtilization', 'Average', 'cpu_utilization'),
            ('EngineCPUUtilization', 'Average', 'engine_cpu'),
            ('DatabaseMemoryUsagePercentage', 'Average', 'memory_usage'),
            ('CacheMisses', 'Sum', 'cache_misses'),
            ('CacheHits', 'Sum', 'cache_hits'),
            ('Evictions', 'Sum', 'evictions'),
            ('CurrConnections', 'Average', 'connections'),
            ('NetworkBytesIn', 'Sum', 'network_in'),
            ('NetworkBytesOut', 'Sum', 'network_out'),
            ('ReplicationLag', 'Average', 'replication_lag'),
            ('BytesUsedForCache', 'Average', 'bytes_used')
        ]
        
        for metric_name, stat, key in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/ElastiCache',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'CacheClusterId', 'Value': cluster_id}
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
            except ClientError:
                pass
        
        # Calculate cache hit ratio
        if 'cache_hits' in metrics and 'cache_misses' in metrics:
            total_requests = metrics['cache_hits']['average'] + metrics['cache_misses']['average']
            if total_requests > 0:
                metrics['cache_hit_ratio'] = (metrics['cache_hits']['average'] / total_requests) * 100
        
        return metrics
    
    def _get_memcached_metrics(self, cluster_id: str, start_time: datetime, 
                               end_time: datetime) -> Dict[str, Any]:
        """Get Memcached-specific metrics."""
        metrics = {}
        
        metric_queries = [
            ('CPUUtilization', 'Average', 'cpu_utilization'),
            ('SwapUsage', 'Average', 'swap_usage'),
            ('BytesUsedForCacheItems', 'Average', 'bytes_used'),
            ('CacheMisses', 'Sum', 'cache_misses'),
            ('CacheHits', 'Sum', 'cache_hits'),
            ('Evictions', 'Sum', 'evictions'),
            ('CurrConnections', 'Average', 'connections'),
            ('NetworkBytesIn', 'Sum', 'network_in'),
            ('NetworkBytesOut', 'Sum', 'network_out')
        ]
        
        for metric_name, stat, key in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/ElastiCache',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'CacheClusterId', 'Value': cluster_id}
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
            except ClientError:
                pass
        
        # Calculate cache hit ratio
        if 'cache_hits' in metrics and 'cache_misses' in metrics:
            total_requests = metrics['cache_hits']['average'] + metrics['cache_misses']['average']
            if total_requests > 0:
                metrics['cache_hit_ratio'] = (metrics['cache_hits']['average'] / total_requests) * 100
        
        return metrics
    
    def _get_replication_group_metrics(self, group_id: str, start_time: datetime,
                                      end_time: datetime) -> Dict[str, Any]:
        """Get replication group metrics."""
        metrics = {}
        
        metric_queries = [
            ('CPUUtilization', 'Average', 'cpu_utilization'),
            ('DatabaseMemoryUsagePercentage', 'Average', 'memory_usage'),
            ('ReplicationLag', 'Average', 'replication_lag')
        ]
        
        for metric_name, stat, key in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/ElastiCache',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'ReplicationGroupId', 'Value': group_id}
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
            except ClientError:
                pass
        
        return metrics
    
    def _analyze_elasticache_usage(self, clusters: List[Dict[str, Any]], 
                                  replication_groups: List[Dict[str, Any]],
                                  snapshots: List[Dict[str, Any]],
                                  metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ElastiCache usage patterns."""
        analysis = {
            'oversized_clusters': [],
            'idle_clusters': [],
            'low_cache_hit_ratio': [],
            'high_evictions': [],
            'old_snapshots': [],
            'unencrypted_clusters': [],
            'single_az_clusters': [],
            'old_engine_versions': [],
            'high_memory_usage': [],
            'replication_lag_issues': []
        }
        
        # Analyze clusters
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            cluster_metrics = metrics.get(cluster_id, {})
            
            # Check for oversized clusters
            cpu_avg = cluster_metrics.get('cpu_utilization', {}).get('average', 0)
            if cpu_avg < 20:
                analysis['oversized_clusters'].append({
                    'cluster_id': cluster_id,
                    'node_type': cluster['cache_node_type'],
                    'cpu_utilization': cpu_avg,
                    'engine': cluster['engine']
                })
            
            # Check for idle clusters
            connections = cluster_metrics.get('connections', {}).get('average', 0)
            if connections < 5:
                analysis['idle_clusters'].append({
                    'cluster_id': cluster_id,
                    'node_type': cluster['cache_node_type'],
                    'avg_connections': connections
                })
            
            # Check cache hit ratio
            cache_hit_ratio = cluster_metrics.get('cache_hit_ratio', 100)
            if cache_hit_ratio < 80:
                analysis['low_cache_hit_ratio'].append({
                    'cluster_id': cluster_id,
                    'hit_ratio': cache_hit_ratio,
                    'engine': cluster['engine']
                })
            
            # Check evictions
            evictions = cluster_metrics.get('evictions', {}).get('average', 0)
            if evictions > 100:
                analysis['high_evictions'].append({
                    'cluster_id': cluster_id,
                    'evictions_per_hour': evictions,
                    'node_type': cluster['cache_node_type']
                })
            
            # Check encryption (Redis only)
            if cluster['engine'] == 'redis' and not cluster.get('replication_group_id'):
                analysis['unencrypted_clusters'].append({
                    'cluster_id': cluster_id,
                    'engine': cluster['engine']
                })
            
            # Check Multi-AZ
            if cluster['num_cache_nodes'] == 1 and 'prod' in cluster_id.lower():
                analysis['single_az_clusters'].append({
                    'cluster_id': cluster_id,
                    'engine': cluster['engine']
                })
            
            # Check engine version
            if self._is_old_engine_version(cluster['engine'], cluster['engine_version']):
                analysis['old_engine_versions'].append({
                    'cluster_id': cluster_id,
                    'engine': cluster['engine'],
                    'current_version': cluster['engine_version']
                })
            
            # Check memory usage (Redis only)
            if cluster['engine'] == 'redis':
                memory_usage = cluster_metrics.get('memory_usage', {}).get('average', 0)
                if memory_usage > 90:
                    analysis['high_memory_usage'].append({
                        'cluster_id': cluster_id,
                        'memory_usage': memory_usage,
                        'node_type': cluster['cache_node_type']
                    })
        
        # Analyze replication groups
        for group in replication_groups:
            group_id = group['replication_group_id']
            group_metrics = metrics.get(f"rg_{group_id}", {})
            
            # Check replication lag
            lag = group_metrics.get('replication_lag', {}).get('average', 0)
            if lag > 1:  # More than 1 second lag
                analysis['replication_lag_issues'].append({
                    'replication_group_id': group_id,
                    'lag_seconds': lag
                })
            
            # Check encryption
            if not group['transit_encryption'] or not group['at_rest_encryption']:
                analysis['unencrypted_clusters'].append({
                    'cluster_id': group_id,
                    'type': 'replication_group',
                    'transit_encryption': group['transit_encryption'],
                    'at_rest_encryption': group['at_rest_encryption']
                })
        
        # Analyze snapshots
        for snapshot in snapshots:
            if snapshot['age_days'] > 90:
                analysis['old_snapshots'].append({
                    'snapshot_name': snapshot['snapshot_name'],
                    'cluster_id': snapshot.get('cluster_id'),
                    'age_days': snapshot['age_days']
                })
        
        return analysis
    
    def _is_old_engine_version(self, engine: str, version: str) -> bool:
        """Check if engine version is old."""
        old_versions = {
            'redis': ['2.', '3.', '4.'],
            'memcached': ['1.4.', '1.5.']
        }
        
        for old_ver in old_versions.get(engine, []):
            if version.startswith(old_ver):
                return True
        
        return False
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ElastiCache optimization recommendations."""
        recommendations = []
        
        # Oversized clusters
        for cluster in analysis['oversized_clusters']:
            current_type = cluster['node_type']
            suggested_type = self._suggest_smaller_node_type(current_type)
            savings = self._estimate_node_type_savings(current_type, suggested_type)
            
            recommendations.append({
                'type': 'elasticache_rightsizing',
                'resource_id': cluster['cluster_id'],
                'title': f"Rightsize ElastiCache cluster: {cluster['cluster_id']}",
                'description': f"CPU utilization is only {cluster['cpu_utilization']:.1f}%. Consider downsizing from {current_type} to {suggested_type}",
                'estimated_savings': savings,
                'priority': 'high',
                'risk_level': 'medium'
            })
        
        # Idle clusters
        for cluster in analysis['idle_clusters']:
            cost = self._estimate_node_type_cost(cluster['node_type'])
            
            recommendations.append({
                'type': 'elasticache_idle',
                'resource_id': cluster['cluster_id'],
                'title': f"Remove idle cluster: {cluster['cluster_id']}",
                'description': f"Cluster has only {cluster['avg_connections']:.1f} average connections. Consider removing.",
                'estimated_savings': cost,
                'priority': 'high',
                'risk_level': 'low'
            })
        
        # Low cache hit ratio
        for cluster in analysis['low_cache_hit_ratio']:
            recommendations.append({
                'type': 'elasticache_hit_ratio',
                'resource_id': cluster['cluster_id'],
                'title': f"Improve cache hit ratio: {cluster['cluster_id']}",
                'description': f"Cache hit ratio is {cluster['hit_ratio']:.1f}%. Review caching strategy or increase cache size.",
                'estimated_savings': 0,
                'priority': 'medium',
                'risk_level': 'low'
            })
        
        # High evictions
        for cluster in analysis['high_evictions']:
            recommendations.append({
                'type': 'elasticache_evictions',
                'resource_id': cluster['cluster_id'],
                'title': f"High evictions detected: {cluster['cluster_id']}",
                'description': f"{cluster['evictions_per_hour']:.0f} evictions/hour. Consider increasing cache size.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'medium'
            })
        
        # Old snapshots
        for snapshot in analysis['old_snapshots']:
            recommendations.append({
                'type': 'elasticache_old_snapshot',
                'resource_id': snapshot['snapshot_name'],
                'title': f"Delete old snapshot: {snapshot['snapshot_name']}",
                'description': f"Snapshot is {snapshot['age_days']} days old. Consider deleting.",
                'estimated_savings': 5,  # Estimate
                'priority': 'low',
                'risk_level': 'low'
            })
        
        # Unencrypted clusters
        for cluster in analysis['unencrypted_clusters']:
            recommendations.append({
                'type': 'elasticache_unencrypted',
                'resource_id': cluster['cluster_id'],
                'title': f"Enable encryption: {cluster['cluster_id']}",
                'description': "Cluster is not encrypted. Enable encryption for security compliance.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'zero'
            })
        
        # High memory usage
        for cluster in analysis['high_memory_usage']:
            recommendations.append({
                'type': 'elasticache_memory',
                'resource_id': cluster['cluster_id'],
                'title': f"High memory usage: {cluster['cluster_id']}",
                'description': f"Memory usage is {cluster['memory_usage']:.1f}%. Consider upgrading node type.",
                'estimated_savings': 0,
                'priority': 'high',
                'risk_level': 'high'
            })
        
        return recommendations
    
    def _suggest_smaller_node_type(self, current_type: str) -> str:
        """Suggest a smaller node type."""
        downsizing_map = {
            'cache.r6g.16xlarge': 'cache.r6g.12xlarge',
            'cache.r6g.12xlarge': 'cache.r6g.8xlarge',
            'cache.r6g.8xlarge': 'cache.r6g.4xlarge',
            'cache.r6g.4xlarge': 'cache.r6g.2xlarge',
            'cache.r6g.2xlarge': 'cache.r6g.xlarge',
            'cache.r6g.xlarge': 'cache.r6g.large',
            'cache.r6g.large': 'cache.t3.medium',
            'cache.m6g.16xlarge': 'cache.m6g.12xlarge',
            'cache.m6g.12xlarge': 'cache.m6g.8xlarge',
            'cache.m6g.8xlarge': 'cache.m6g.4xlarge',
            'cache.m6g.4xlarge': 'cache.m6g.2xlarge',
            'cache.m6g.2xlarge': 'cache.m6g.xlarge',
            'cache.m6g.xlarge': 'cache.m6g.large',
            'cache.m6g.large': 'cache.t3.small'
        }
        
        return downsizing_map.get(current_type, 'cache.t3.micro')
    
    def _estimate_node_type_savings(self, current_type: str, suggested_type: str) -> float:
        """Estimate savings from node type change."""
        monthly_costs = {
            'cache.r6g.16xlarge': 5800,
            'cache.r6g.12xlarge': 4350,
            'cache.r6g.8xlarge': 2900,
            'cache.r6g.4xlarge': 1450,
            'cache.r6g.2xlarge': 725,
            'cache.r6g.xlarge': 362,
            'cache.r6g.large': 181,
            'cache.t3.medium': 52,
            'cache.t3.small': 26,
            'cache.t3.micro': 13
        }
        
        current_cost = monthly_costs.get(current_type, 200)
        suggested_cost = monthly_costs.get(suggested_type, 100)
        
        return max(0, current_cost - suggested_cost)
    
    def _estimate_node_type_cost(self, node_type: str) -> float:
        """Estimate monthly cost of a node type."""
        monthly_costs = {
            'cache.r6g.16xlarge': 5800,
            'cache.r6g.12xlarge': 4350,
            'cache.r6g.8xlarge': 2900,
            'cache.r6g.4xlarge': 1450,
            'cache.r6g.2xlarge': 725,
            'cache.r6g.xlarge': 362,
            'cache.r6g.large': 181,
            'cache.t3.medium': 52,
            'cache.t3.small': 26,
            'cache.t3.micro': 13
        }
        
        return monthly_costs.get(node_type, 100)
    
    def _generate_summary(self, analysis: Dict[str, Any], 
                         recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate ElastiCache optimization summary."""
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        
        return {
            'total_clusters': sum(len(v) for v in analysis.values()),
            'optimization_opportunities': len(recommendations),
            'potential_monthly_savings': total_savings,
            'oversized_count': len(analysis['oversized_clusters']),
            'idle_count': len(analysis['idle_clusters']),
            'low_hit_ratio_count': len(analysis['low_cache_hit_ratio']),
            'high_evictions_count': len(analysis['high_evictions']),
            'unencrypted_count': len(analysis['unencrypted_clusters']),
            'old_snapshots_count': len(analysis['old_snapshots'])
        }