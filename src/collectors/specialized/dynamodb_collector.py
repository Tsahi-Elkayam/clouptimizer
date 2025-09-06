"""
AWS DynamoDB Specialized Collector
Comprehensive DynamoDB table analysis including capacity, backups, and optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
import concurrent.futures

logger = logging.getLogger(__name__)


class DynamoDBCollector:
    """
    Specialized collector for AWS DynamoDB tables.
    Analyzes capacity, usage patterns, and optimization opportunities.
    """
    
    # DynamoDB pricing (simplified - varies by region)
    PRICING = {
        'on_demand_read': 0.00000025,  # Per read request unit
        'on_demand_write': 0.00000125,  # Per write request unit
        'provisioned_read_hour': 0.00013,  # Per RCU per hour
        'provisioned_write_hour': 0.00065,  # Per WCU per hour
        'storage_gb': 0.25,  # Per GB per month
        'backup_gb': 0.10,  # Per GB per month for backups
        'global_table_replica': 1.5,  # Multiplier for global tables
        'streams_request': 0.02,  # Per 100,000 requests
    }
    
    def __init__(self, session, regions: List[str] = None):
        """
        Initialize DynamoDB collector.
        
        Args:
            session: Boto3 session
            regions: List of regions to collect from
        """
        self.session = session
        self.regions = regions or ['us-east-1']
        self.tables = []
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect DynamoDB table data from all regions.
        
        Returns:
            List of DynamoDB table resources with metrics
        """
        self.tables = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for region in self.regions:
                futures.append(
                    executor.submit(self._collect_region, region)
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    region_tables = future.result()
                    self.tables.extend(region_tables)
                except Exception as e:
                    logger.error(f"Error collecting DynamoDB data: {e}")
        
        return self.tables
    
    def _collect_region(self, region: str) -> List[Dict]:
        """Collect DynamoDB tables from a specific region"""
        tables = []
        
        try:
            dynamodb = self.session.client('dynamodb', region_name=region)
            cloudwatch = self.session.client('cloudwatch', region_name=region)
            
            # List all tables
            paginator = dynamodb.get_paginator('list_tables')
            
            for page in paginator.paginate():
                for table_name in page.get('TableNames', []):
                    table_data = self._analyze_table(
                        table_name, dynamodb, cloudwatch, region
                    )
                    if table_data:
                        tables.append(table_data)
            
            logger.info(f"Collected {len(tables)} DynamoDB tables from {region}")
            
        except ClientError as e:
            logger.error(f"Error collecting DynamoDB tables from {region}: {e}")
        
        return tables
    
    def _analyze_table(self, table_name: str, dynamodb, cloudwatch, region: str) -> Dict:
        """Analyze individual DynamoDB table"""
        try:
            # Get table description
            table = dynamodb.describe_table(TableName=table_name)['Table']
            
            # Get table metrics
            metrics = self._get_table_metrics(table_name, cloudwatch)
            
            # Get backup information
            backup_info = self._get_backup_info(table_name, dynamodb)
            
            # Calculate costs
            costs = self._calculate_costs(table, metrics, backup_info)
            
            # Analyze capacity and usage
            capacity_analysis = self._analyze_capacity(table, metrics)
            
            # Identify optimization opportunities
            optimizations = self._identify_optimizations(
                table, metrics, capacity_analysis, costs
            )
            
            return {
                'id': table['TableArn'],
                'type': 'dynamodb_table',
                'name': table_name,
                'region': region,
                'status': table['TableStatus'],
                'billing_mode': table.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED'),
                'size_bytes': table.get('TableSizeBytes', 0),
                'item_count': table.get('ItemCount', 0),
                'creation_time': table['CreationDateTime'].isoformat() if table.get('CreationDateTime') else None,
                'provisioned_throughput': {
                    'read_capacity': table.get('ProvisionedThroughput', {}).get('ReadCapacityUnits', 0),
                    'write_capacity': table.get('ProvisionedThroughput', {}).get('WriteCapacityUnits', 0)
                },
                'global_secondary_indexes': len(table.get('GlobalSecondaryIndexes', [])),
                'local_secondary_indexes': len(table.get('LocalSecondaryIndexes', [])),
                'stream_enabled': table.get('StreamSpecification', {}).get('StreamEnabled', False),
                'global_table': len(table.get('Replicas', [])) > 0,
                'encryption': table.get('SSEDescription', {}).get('Status', 'DISABLED'),
                'point_in_time_recovery': table.get('PointInTimeRecoveryDescription', {}).get('PointInTimeRecoveryStatus', 'DISABLED'),
                'metrics': metrics,
                'backup_info': backup_info,
                'costs': costs,
                'capacity_analysis': capacity_analysis,
                'optimizations': optimizations,
                'tags': self._get_table_tags(table['TableArn'], dynamodb),
                'metadata': {
                    'table_class': table.get('TableClassSummary', {}).get('TableClass', 'STANDARD'),
                    'deletion_protection': table.get('DeletionProtectionEnabled', False),
                    'ttl_enabled': self._check_ttl_status(table_name, dynamodb),
                    'contributor_insights': self._check_contributor_insights(table_name, dynamodb)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {e}")
            return None
    
    def _get_table_metrics(self, table_name: str, cloudwatch) -> Dict:
        """Get CloudWatch metrics for DynamoDB table"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        metrics = {
            'consumed_read_capacity': 0,
            'consumed_write_capacity': 0,
            'throttled_requests': 0,
            'user_errors': 0,
            'system_errors': 0,
            'successful_requests': 0,
            'average_latency': 0,
            'account_provisioned_read': 0,
            'account_provisioned_write': 0
        }
        
        try:
            # Get consumed read capacity
            read_capacity = cloudwatch.get_metric_statistics(
                Namespace='AWS/DynamoDB',
                MetricName='ConsumedReadCapacityUnits',
                Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # Hourly
                Statistics=['Sum', 'Average']
            )
            
            if read_capacity['Datapoints']:
                metrics['consumed_read_capacity'] = sum(dp['Sum'] for dp in read_capacity['Datapoints'])
            
            # Get consumed write capacity
            write_capacity = cloudwatch.get_metric_statistics(
                Namespace='AWS/DynamoDB',
                MetricName='ConsumedWriteCapacityUnits',
                Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum', 'Average']
            )
            
            if write_capacity['Datapoints']:
                metrics['consumed_write_capacity'] = sum(dp['Sum'] for dp in write_capacity['Datapoints'])
            
            # Get throttled requests
            throttled = cloudwatch.get_metric_statistics(
                Namespace='AWS/DynamoDB',
                MetricName='UserErrors',
                Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # Daily
                Statistics=['Sum']
            )
            
            if throttled['Datapoints']:
                metrics['throttled_requests'] = sum(dp['Sum'] for dp in throttled['Datapoints'])
            
            # Get successful request latency
            latency = cloudwatch.get_metric_statistics(
                Namespace='AWS/DynamoDB',
                MetricName='SuccessfulRequestLatency',
                Dimensions=[
                    {'Name': 'TableName', 'Value': table_name},
                    {'Name': 'Operation', 'Value': 'GetItem'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if latency['Datapoints']:
                metrics['average_latency'] = sum(dp['Average'] for dp in latency['Datapoints']) / len(latency['Datapoints'])
            
        except Exception as e:
            logger.error(f"Error getting metrics for {table_name}: {e}")
        
        return metrics
    
    def _get_backup_info(self, table_name: str, dynamodb) -> Dict:
        """Get backup information for table"""
        backup_info = {
            'continuous_backups_enabled': False,
            'backup_count': 0,
            'total_backup_size_bytes': 0,
            'latest_backup_time': None
        }
        
        try:
            # Check continuous backups
            continuous = dynamodb.describe_continuous_backups(TableName=table_name)
            backup_info['continuous_backups_enabled'] = (
                continuous['ContinuousBackupsDescription']['ContinuousBackupsStatus'] == 'ENABLED'
            )
            
            # List on-demand backups
            backups = dynamodb.list_backups(TableName=table_name)
            backup_info['backup_count'] = len(backups.get('BackupSummaries', []))
            
            if backups.get('BackupSummaries'):
                backup_info['total_backup_size_bytes'] = sum(
                    b.get('BackupSizeBytes', 0) for b in backups['BackupSummaries']
                )
                latest = max(backups['BackupSummaries'], key=lambda x: x.get('BackupCreationDateTime', datetime.min))
                backup_info['latest_backup_time'] = latest.get('BackupCreationDateTime', '').isoformat() if latest.get('BackupCreationDateTime') else None
                
        except Exception as e:
            logger.debug(f"Could not get backup info for {table_name}: {e}")
        
        return backup_info
    
    def _calculate_costs(self, table: Dict, metrics: Dict, backup_info: Dict) -> Dict:
        """Calculate DynamoDB table costs"""
        billing_mode = table.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED')
        size_gb = table.get('TableSizeBytes', 0) / (1024**3)
        
        costs = {
            'monthly_storage_cost': size_gb * self.PRICING['storage_gb'],
            'monthly_capacity_cost': 0,
            'monthly_backup_cost': 0,
            'monthly_total_cost': 0,
            'annual_cost': 0
        }
        
        # Calculate capacity costs
        if billing_mode == 'PAY_PER_REQUEST':
            # On-demand pricing
            read_requests = metrics['consumed_read_capacity']
            write_requests = metrics['consumed_write_capacity']
            costs['monthly_capacity_cost'] = (
                read_requests * self.PRICING['on_demand_read'] +
                write_requests * self.PRICING['on_demand_write']
            )
        else:
            # Provisioned capacity pricing
            read_capacity = table.get('ProvisionedThroughput', {}).get('ReadCapacityUnits', 0)
            write_capacity = table.get('ProvisionedThroughput', {}).get('WriteCapacityUnits', 0)
            
            # Add GSI capacity
            for gsi in table.get('GlobalSecondaryIndexes', []):
                read_capacity += gsi.get('ProvisionedThroughput', {}).get('ReadCapacityUnits', 0)
                write_capacity += gsi.get('ProvisionedThroughput', {}).get('WriteCapacityUnits', 0)
            
            costs['monthly_capacity_cost'] = (
                read_capacity * self.PRICING['provisioned_read_hour'] * 730 +
                write_capacity * self.PRICING['provisioned_write_hour'] * 730
            )
        
        # Calculate backup costs
        if backup_info['continuous_backups_enabled']:
            costs['monthly_backup_cost'] = size_gb * self.PRICING['backup_gb']
        
        backup_size_gb = backup_info['total_backup_size_bytes'] / (1024**3)
        costs['monthly_backup_cost'] += backup_size_gb * self.PRICING['backup_gb']
        
        # Global table multiplier
        if len(table.get('Replicas', [])) > 0:
            replica_count = len(table.get('Replicas', []))
            costs['monthly_capacity_cost'] *= (1 + replica_count)
            costs['monthly_storage_cost'] *= (1 + replica_count)
        
        costs['monthly_total_cost'] = (
            costs['monthly_storage_cost'] +
            costs['monthly_capacity_cost'] +
            costs['monthly_backup_cost']
        )
        costs['annual_cost'] = costs['monthly_total_cost'] * 12
        
        return costs
    
    def _analyze_capacity(self, table: Dict, metrics: Dict) -> Dict:
        """Analyze table capacity utilization"""
        billing_mode = table.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED')
        
        analysis = {
            'billing_mode': billing_mode,
            'read_utilization': 0,
            'write_utilization': 0,
            'is_overprovisioned': False,
            'is_underprovisioned': False,
            'throttling_detected': False,
            'recommended_read_capacity': 0,
            'recommended_write_capacity': 0
        }
        
        if billing_mode == 'PROVISIONED':
            read_capacity = table.get('ProvisionedThroughput', {}).get('ReadCapacityUnits', 1)
            write_capacity = table.get('ProvisionedThroughput', {}).get('WriteCapacityUnits', 1)
            
            # Calculate utilization (monthly average)
            hours_in_month = 730
            avg_read_consumed = metrics['consumed_read_capacity'] / hours_in_month
            avg_write_consumed = metrics['consumed_write_capacity'] / hours_in_month
            
            analysis['read_utilization'] = (avg_read_consumed / read_capacity * 100) if read_capacity > 0 else 0
            analysis['write_utilization'] = (avg_write_consumed / write_capacity * 100) if write_capacity > 0 else 0
            
            # Check for over/under provisioning
            if analysis['read_utilization'] < 20 or analysis['write_utilization'] < 20:
                analysis['is_overprovisioned'] = True
                analysis['recommended_read_capacity'] = max(1, int(avg_read_consumed * 1.5))
                analysis['recommended_write_capacity'] = max(1, int(avg_write_consumed * 1.5))
            elif analysis['read_utilization'] > 80 or analysis['write_utilization'] > 80:
                analysis['is_underprovisioned'] = True
                analysis['recommended_read_capacity'] = int(avg_read_consumed * 1.5)
                analysis['recommended_write_capacity'] = int(avg_write_consumed * 1.5)
            
            # Check for throttling
            if metrics['throttled_requests'] > 0:
                analysis['throttling_detected'] = True
        
        return analysis
    
    def _identify_optimizations(self, table: Dict, metrics: Dict, 
                               capacity_analysis: Dict, costs: Dict) -> List[Dict]:
        """Identify optimization opportunities for DynamoDB table"""
        optimizations = []
        billing_mode = table.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED')
        
        # Check for unused tables (no requests in 30 days)
        if metrics['consumed_read_capacity'] == 0 and metrics['consumed_write_capacity'] == 0:
            optimizations.append({
                'type': 'unused_table',
                'severity': 'high',
                'description': 'Table has no read/write activity in 30 days',
                'recommendation': 'Consider deleting or archiving this table',
                'potential_savings': costs['monthly_total_cost'],
                'effort': 'low',
                'risk': 'medium'
            })
        
        # Check for overprovisioned capacity
        elif capacity_analysis['is_overprovisioned'] and billing_mode == 'PROVISIONED':
            current_capacity_cost = costs['monthly_capacity_cost']
            new_read = capacity_analysis['recommended_read_capacity']
            new_write = capacity_analysis['recommended_write_capacity']
            
            new_cost = (
                new_read * self.PRICING['provisioned_read_hour'] * 730 +
                new_write * self.PRICING['provisioned_write_hour'] * 730
            )
            
            savings = current_capacity_cost - new_cost
            
            if savings > 10:  # Minimum $10 savings
                optimizations.append({
                    'type': 'overprovisioned_capacity',
                    'severity': 'medium',
                    'description': f"Table capacity utilization is low (Read: {capacity_analysis['read_utilization']:.1f}%, Write: {capacity_analysis['write_utilization']:.1f}%)",
                    'recommendation': f"Reduce capacity or switch to on-demand mode",
                    'potential_savings': savings,
                    'effort': 'low',
                    'risk': 'low'
                })
        
        # Check for throttling
        if capacity_analysis['throttling_detected']:
            optimizations.append({
                'type': 'throttling',
                'severity': 'high',
                'description': f"Table experienced {metrics['throttled_requests']} throttled requests",
                'recommendation': 'Increase provisioned capacity or switch to on-demand',
                'potential_savings': 0,  # This is about availability
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for missing TTL
        if not table.get('metadata', {}).get('ttl_enabled') and table.get('ItemCount', 0) > 10000:
            # Estimate 20% of items could be expired
            potential_storage_reduction = costs['monthly_storage_cost'] * 0.2
            optimizations.append({
                'type': 'missing_ttl',
                'severity': 'low',
                'description': 'TTL is not enabled for automatic item expiration',
                'recommendation': 'Enable TTL to automatically delete expired items',
                'potential_savings': potential_storage_reduction,
                'effort': 'medium',
                'risk': 'low'
            })
        
        # Check for Standard-IA table class opportunity
        size_gb = table.get('TableSizeBytes', 0) / (1024**3)
        if size_gb > 100 and capacity_analysis['read_utilization'] < 30:
            # Standard-IA offers 60% storage cost reduction
            ia_savings = costs['monthly_storage_cost'] * 0.6
            optimizations.append({
                'type': 'table_class_optimization',
                'severity': 'medium',
                'description': f"Large table ({size_gb:.1f}GB) with low access frequency",
                'recommendation': 'Consider DynamoDB Standard-IA table class',
                'potential_savings': ia_savings,
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for on-demand vs provisioned optimization
        if billing_mode == 'PROVISIONED' and capacity_analysis['read_utilization'] < 10:
            optimizations.append({
                'type': 'billing_mode_optimization',
                'severity': 'medium',
                'description': 'Very low and unpredictable usage pattern',
                'recommendation': 'Consider switching to on-demand billing',
                'potential_savings': costs['monthly_capacity_cost'] * 0.3,  # Estimate
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for excessive backups
        if table.get('backup_info', {}).get('backup_count', 0) > 30:
            optimizations.append({
                'type': 'excessive_backups',
                'severity': 'low',
                'description': f"{table.get('backup_info', {}).get('backup_count')} on-demand backups retained",
                'recommendation': 'Review and delete old backups',
                'potential_savings': costs['monthly_backup_cost'] * 0.5,
                'effort': 'low',
                'risk': 'low'
            })
        
        return optimizations
    
    def _get_table_tags(self, table_arn: str, dynamodb) -> Dict:
        """Get tags for DynamoDB table"""
        try:
            response = dynamodb.list_tags_of_resource(ResourceArn=table_arn)
            return {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
        except:
            return {}
    
    def _check_ttl_status(self, table_name: str, dynamodb) -> bool:
        """Check if TTL is enabled"""
        try:
            response = dynamodb.describe_time_to_live(TableName=table_name)
            return response['TimeToLiveDescription']['TimeToLiveStatus'] == 'ENABLED'
        except:
            return False
    
    def _check_contributor_insights(self, table_name: str, dynamodb) -> str:
        """Check Contributor Insights status"""
        try:
            response = dynamodb.describe_contributor_insights(TableName=table_name)
            return response.get('ContributorInsightsStatus', 'DISABLED')
        except:
            return 'DISABLED'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of DynamoDB collection"""
        if not self.tables:
            return {}
        
        total_tables = len(self.tables)
        total_cost = sum(t['costs']['monthly_total_cost'] for t in self.tables)
        total_storage_gb = sum(t['size_bytes'] / (1024**3) for t in self.tables)
        
        unused_tables = [t for t in self.tables 
                        if t['metrics']['consumed_read_capacity'] == 0 
                        and t['metrics']['consumed_write_capacity'] == 0]
        
        overprovisioned = [t for t in self.tables 
                          if t['capacity_analysis']['is_overprovisioned']]
        
        throttled = [t for t in self.tables 
                    if t['capacity_analysis']['throttling_detected']]
        
        return {
            'total_tables': total_tables,
            'total_monthly_cost': total_cost,
            'total_storage_gb': total_storage_gb,
            'unused_tables': len(unused_tables),
            'overprovisioned_tables': len(overprovisioned),
            'throttled_tables': len(throttled),
            'potential_savings': sum(
                sum(opt['potential_savings'] for opt in t['optimizations'])
                for t in self.tables
            ),
            'by_billing_mode': self._group_by_billing_mode(),
            'by_region': self._group_by_region(),
            'top_cost_tables': self._get_top_cost_tables(5)
        }
    
    def _group_by_billing_mode(self) -> Dict:
        """Group tables by billing mode"""
        by_mode = defaultdict(lambda: {'count': 0, 'cost': 0})
        for table in self.tables:
            mode = table.get('billing_mode', 'PROVISIONED')
            by_mode[mode]['count'] += 1
            by_mode[mode]['cost'] += table['costs']['monthly_total_cost']
        return dict(by_mode)
    
    def _group_by_region(self) -> Dict:
        """Group tables by region"""
        by_region = defaultdict(lambda: {'count': 0, 'cost': 0})
        for table in self.tables:
            region = table.get('region', 'unknown')
            by_region[region]['count'] += 1
            by_region[region]['cost'] += table['costs']['monthly_total_cost']
        return dict(by_region)
    
    def _get_top_cost_tables(self, limit: int = 5) -> List[Dict]:
        """Get top cost tables"""
        sorted_tables = sorted(
            self.tables,
            key=lambda x: x['costs']['monthly_total_cost'],
            reverse=True
        )
        
        return [
            {
                'name': t['name'],
                'region': t['region'],
                'monthly_cost': t['costs']['monthly_total_cost'],
                'size_gb': t['size_bytes'] / (1024**3),
                'billing_mode': t['billing_mode']
            }
            for t in sorted_tables[:limit]
        ]