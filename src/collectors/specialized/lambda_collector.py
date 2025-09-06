"""
AWS Lambda Specialized Collector
Comprehensive Lambda function analysis including cold starts, memory optimization, and dead functions.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
import concurrent.futures
import time

logger = logging.getLogger(__name__)


class LambdaCollector:
    """
    Specialized collector for AWS Lambda functions.
    Analyzes function usage, performance, and optimization opportunities.
    """
    
    # Lambda pricing (simplified - varies by region)
    PRICING = {
        'request_price': 0.0000002,  # Per request
        'gb_second_price': 0.0000166667,  # Per GB-second
        'free_tier_requests': 1000000,  # Monthly free tier
        'free_tier_gb_seconds': 400000,  # Monthly free tier
    }
    
    def __init__(self, session, regions: List[str] = None):
        """
        Initialize Lambda collector.
        
        Args:
            session: Boto3 session
            regions: List of regions to collect from
        """
        self.session = session
        self.regions = regions or ['us-east-1']
        self.functions = []
        self.metrics = {}
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect Lambda function data from all regions.
        
        Returns:
            List of Lambda function resources with metrics
        """
        self.functions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for region in self.regions:
                futures.append(
                    executor.submit(self._collect_region, region)
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    region_functions = future.result()
                    self.functions.extend(region_functions)
                except Exception as e:
                    logger.error(f"Error collecting Lambda data: {e}")
        
        return self.functions
    
    def _collect_region(self, region: str) -> List[Dict]:
        """Collect Lambda functions from a specific region"""
        functions = []
        
        try:
            lambda_client = self.session.client('lambda', region_name=region)
            cloudwatch = self.session.client('cloudwatch', region_name=region)
            logs_client = self.session.client('logs', region_name=region)
            
            # List all functions
            paginator = lambda_client.get_paginator('list_functions')
            
            for page in paginator.paginate():
                for func in page.get('Functions', []):
                    function_data = self._analyze_function(
                        func, lambda_client, cloudwatch, logs_client, region
                    )
                    if function_data:
                        functions.append(function_data)
            
            logger.info(f"Collected {len(functions)} Lambda functions from {region}")
            
        except ClientError as e:
            logger.error(f"Error collecting Lambda functions from {region}: {e}")
        
        return functions
    
    def _analyze_function(self, function: Dict, lambda_client, cloudwatch, 
                         logs_client, region: str) -> Dict:
        """Analyze individual Lambda function"""
        function_name = function['FunctionName']
        
        try:
            # Get function configuration
            config = lambda_client.get_function_configuration(
                FunctionName=function_name
            )
            
            # Get function metrics
            metrics = self._get_function_metrics(function_name, cloudwatch)
            
            # Get function logs metrics
            log_metrics = self._get_log_metrics(function_name, logs_client)
            
            # Calculate costs
            costs = self._calculate_costs(metrics, config)
            
            # Analyze performance
            performance = self._analyze_performance(metrics, config)
            
            # Identify optimization opportunities
            optimizations = self._identify_optimizations(
                config, metrics, performance, costs
            )
            
            return {
                'id': function['FunctionArn'],
                'type': 'lambda_function',
                'name': function_name,
                'region': region,
                'runtime': function.get('Runtime', 'unknown'),
                'memory_size': config.get('MemorySize', 128),
                'timeout': config.get('Timeout', 3),
                'code_size': function.get('CodeSize', 0),
                'last_modified': function.get('LastModified'),
                'environment_variables': len(config.get('Environment', {}).get('Variables', {})),
                'layers': len(config.get('Layers', [])),
                'reserved_concurrent_executions': config.get('ReservedConcurrentExecutions'),
                'metrics': metrics,
                'performance': performance,
                'costs': costs,
                'log_metrics': log_metrics,
                'optimizations': optimizations,
                'tags': self._get_function_tags(function['FunctionArn'], lambda_client),
                'metadata': {
                    'handler': config.get('Handler'),
                    'description': config.get('Description', ''),
                    'architecture': config.get('Architectures', ['x86_64'])[0],
                    'ephemeral_storage': config.get('EphemeralStorage', {}).get('Size', 512),
                    'state': config.get('State', 'Active'),
                    'state_reason': config.get('StateReason'),
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing function {function_name}: {e}")
            return None
    
    def _get_function_metrics(self, function_name: str, cloudwatch) -> Dict:
        """Get CloudWatch metrics for Lambda function"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        metrics = {
            'invocations': 0,
            'errors': 0,
            'throttles': 0,
            'duration_avg': 0,
            'duration_max': 0,
            'concurrent_executions_max': 0,
            'cold_starts': 0,
            'memory_utilization': 0,
            'timeout_errors': 0,
            'daily_invocations': []
        }
        
        try:
            # Get invocation count
            invocations = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # Daily
                Statistics=['Sum']
            )
            
            if invocations['Datapoints']:
                metrics['invocations'] = sum(dp['Sum'] for dp in invocations['Datapoints'])
                metrics['daily_invocations'] = [dp['Sum'] for dp in invocations['Datapoints']]
            
            # Get error count
            errors = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Errors',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=2592000,  # 30 days
                Statistics=['Sum']
            )
            
            if errors['Datapoints']:
                metrics['errors'] = sum(dp['Sum'] for dp in errors['Datapoints'])
            
            # Get duration statistics
            duration = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=2592000,
                Statistics=['Average', 'Maximum']
            )
            
            if duration['Datapoints']:
                metrics['duration_avg'] = sum(dp['Average'] for dp in duration['Datapoints']) / len(duration['Datapoints'])
                metrics['duration_max'] = max(dp['Maximum'] for dp in duration['Datapoints'])
            
            # Get concurrent executions
            concurrent = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='ConcurrentExecutions',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Maximum']
            )
            
            if concurrent['Datapoints']:
                metrics['concurrent_executions_max'] = max(dp['Maximum'] for dp in concurrent['Datapoints'])
            
            # Estimate cold starts (simplified)
            # In production, use X-Ray or custom metrics
            if metrics['invocations'] > 0:
                metrics['cold_starts'] = int(metrics['invocations'] * 0.05)  # Estimate 5%
            
        except Exception as e:
            logger.error(f"Error getting metrics for {function_name}: {e}")
        
        return metrics
    
    def _get_log_metrics(self, function_name: str, logs_client) -> Dict:
        """Get CloudWatch Logs metrics for Lambda function"""
        log_group = f"/aws/lambda/{function_name}"
        
        metrics = {
            'log_size_bytes': 0,
            'log_streams': 0,
            'retention_days': None,
            'last_event_time': None
        }
        
        try:
            # Get log group info
            response = logs_client.describe_log_groups(
                logGroupNamePrefix=log_group,
                limit=1
            )
            
            if response.get('logGroups'):
                log_group_data = response['logGroups'][0]
                metrics['log_size_bytes'] = log_group_data.get('storedBytes', 0)
                metrics['retention_days'] = log_group_data.get('retentionInDays')
                
                # Get log streams
                streams = logs_client.describe_log_streams(
                    logGroupName=log_group,
                    orderBy='LastEventTime',
                    descending=True,
                    limit=50
                )
                
                metrics['log_streams'] = len(streams.get('logStreams', []))
                
                if streams.get('logStreams'):
                    last_stream = streams['logStreams'][0]
                    if last_stream.get('lastEventTimestamp'):
                        metrics['last_event_time'] = datetime.fromtimestamp(
                            last_stream['lastEventTimestamp'] / 1000
                        ).isoformat()
                        
        except Exception as e:
            logger.debug(f"Could not get log metrics for {function_name}: {e}")
        
        return metrics
    
    def _calculate_costs(self, metrics: Dict, config: Dict) -> Dict:
        """Calculate Lambda function costs"""
        memory_gb = config.get('MemorySize', 128) / 1024
        
        # Calculate request costs
        total_requests = metrics.get('invocations', 0)
        billable_requests = max(0, total_requests - self.PRICING['free_tier_requests'])
        request_cost = billable_requests * self.PRICING['request_price']
        
        # Calculate compute costs (GB-seconds)
        avg_duration_ms = metrics.get('duration_avg', 0)
        avg_duration_seconds = avg_duration_ms / 1000
        total_gb_seconds = total_requests * avg_duration_seconds * memory_gb
        billable_gb_seconds = max(0, total_gb_seconds - self.PRICING['free_tier_gb_seconds'])
        compute_cost = billable_gb_seconds * self.PRICING['gb_second_price']
        
        # Calculate monthly costs
        monthly_cost = request_cost + compute_cost
        
        # Calculate waste (overprovisioned memory)
        memory_utilization = metrics.get('memory_utilization', 50)  # Default 50%
        if memory_utilization < 50:
            waste_percentage = (50 - memory_utilization) / 50
            wasted_cost = monthly_cost * waste_percentage * 0.5  # Estimate 50% of waste is recoverable
        else:
            wasted_cost = 0
        
        return {
            'monthly_cost': monthly_cost,
            'annual_cost': monthly_cost * 12,
            'request_cost': request_cost,
            'compute_cost': compute_cost,
            'total_gb_seconds': total_gb_seconds,
            'billable_gb_seconds': billable_gb_seconds,
            'wasted_cost': wasted_cost,
            'cost_per_invocation': monthly_cost / total_requests if total_requests > 0 else 0,
            'cost_per_gb_second': self.PRICING['gb_second_price']
        }
    
    def _analyze_performance(self, metrics: Dict, config: Dict) -> Dict:
        """Analyze Lambda function performance"""
        timeout_ms = config.get('Timeout', 3) * 1000
        memory_size = config.get('MemorySize', 128)
        
        performance = {
            'is_timing_out': False,
            'is_throttled': False,
            'cold_start_percentage': 0,
            'error_rate': 0,
            'average_duration_percentage': 0,
            'memory_efficiency': 0,
            'concurrency_utilization': 0
        }
        
        # Check for timeouts
        if metrics['duration_max'] >= timeout_ms * 0.95:  # Within 5% of timeout
            performance['is_timing_out'] = True
        
        # Check for throttling
        if metrics.get('throttles', 0) > 0:
            performance['is_throttled'] = True
        
        # Calculate cold start percentage
        if metrics['invocations'] > 0:
            performance['cold_start_percentage'] = (
                metrics.get('cold_starts', 0) / metrics['invocations'] * 100
            )
        
        # Calculate error rate
        if metrics['invocations'] > 0:
            performance['error_rate'] = (
                metrics.get('errors', 0) / metrics['invocations'] * 100
            )
        
        # Calculate duration percentage of timeout
        if timeout_ms > 0:
            performance['average_duration_percentage'] = (
                metrics['duration_avg'] / timeout_ms * 100
            )
        
        # Estimate memory efficiency
        # In production, use CloudWatch Lambda Insights or X-Ray
        if metrics['duration_avg'] > 0:
            # Simplified: assume 50% memory usage if duration is less than 50% of timeout
            if performance['average_duration_percentage'] < 50:
                performance['memory_efficiency'] = 40  # Likely overprovisioned
            else:
                performance['memory_efficiency'] = 70  # Reasonable
        
        # Calculate concurrency utilization
        reserved = config.get('ReservedConcurrentExecutions')
        if reserved and reserved > 0:
            performance['concurrency_utilization'] = (
                metrics.get('concurrent_executions_max', 0) / reserved * 100
            )
        
        return performance
    
    def _identify_optimizations(self, config: Dict, metrics: Dict, 
                               performance: Dict, costs: Dict) -> List[Dict]:
        """Identify optimization opportunities for Lambda function"""
        optimizations = []
        
        # Check for dead functions (no invocations in 30 days)
        if metrics['invocations'] == 0:
            optimizations.append({
                'type': 'dead_function',
                'severity': 'high',
                'description': 'Function has not been invoked in 30 days',
                'recommendation': 'Consider deleting this function',
                'potential_savings': costs['monthly_cost'],
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for overprovisioned memory
        elif performance['memory_efficiency'] < 50 and config.get('MemorySize', 128) > 128:
            recommended_memory = max(128, config['MemorySize'] // 2)
            new_cost = costs['monthly_cost'] * (recommended_memory / config['MemorySize'])
            savings = costs['monthly_cost'] - new_cost
            
            optimizations.append({
                'type': 'overprovisioned_memory',
                'severity': 'medium',
                'description': f"Function uses less than 50% of allocated memory",
                'recommendation': f"Reduce memory from {config['MemorySize']}MB to {recommended_memory}MB",
                'potential_savings': savings,
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for timeout issues
        if performance['is_timing_out']:
            optimizations.append({
                'type': 'timeout_issues',
                'severity': 'high',
                'description': 'Function is timing out',
                'recommendation': 'Increase timeout or optimize function code',
                'potential_savings': 0,  # No direct savings but prevents errors
                'effort': 'medium',
                'risk': 'medium'
            })
        
        # Check for high error rate
        if performance['error_rate'] > 5:
            optimizations.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'description': f"Error rate is {performance['error_rate']:.1f}%",
                'recommendation': 'Fix function errors to reduce wasted invocations',
                'potential_savings': costs['monthly_cost'] * (performance['error_rate'] / 100),
                'effort': 'high',
                'risk': 'low'
            })
        
        # Check for cold start issues
        if performance['cold_start_percentage'] > 10:
            optimizations.append({
                'type': 'cold_start_issues',
                'severity': 'medium',
                'description': f"Cold starts affect {performance['cold_start_percentage']:.1f}% of invocations",
                'recommendation': 'Consider provisioned concurrency or reduce function size',
                'potential_savings': 0,  # Performance optimization
                'effort': 'medium',
                'risk': 'low'
            })
        
        # Check for throttling
        if performance['is_throttled']:
            optimizations.append({
                'type': 'throttling',
                'severity': 'high',
                'description': 'Function is being throttled',
                'recommendation': 'Increase reserved concurrent executions',
                'potential_savings': 0,  # Availability optimization
                'effort': 'low',
                'risk': 'low'
            })
        
        # Check for old runtimes
        runtime = config.get('Runtime', '')
        if self._is_deprecated_runtime(runtime):
            optimizations.append({
                'type': 'deprecated_runtime',
                'severity': 'high',
                'description': f"Runtime {runtime} is deprecated or outdated",
                'recommendation': 'Update to a supported runtime version',
                'potential_savings': 0,  # Security/maintenance optimization
                'effort': 'medium',
                'risk': 'medium'
            })
        
        # Check for excessive logging
        log_size_gb = metrics.get('log_metrics', {}).get('log_size_bytes', 0) / (1024**3)
        if log_size_gb > 5:
            log_cost = log_size_gb * 0.50  # $0.50 per GB for CloudWatch Logs
            optimizations.append({
                'type': 'excessive_logging',
                'severity': 'low',
                'description': f"Function has {log_size_gb:.1f}GB of logs",
                'recommendation': 'Reduce log verbosity or set retention policy',
                'potential_savings': log_cost * 0.5,  # Assume 50% reduction possible
                'effort': 'low',
                'risk': 'low'
            })
        
        return optimizations
    
    def _get_function_tags(self, function_arn: str, lambda_client) -> Dict:
        """Get tags for Lambda function"""
        try:
            response = lambda_client.list_tags(Resource=function_arn)
            return response.get('Tags', {})
        except:
            return {}
    
    def _is_deprecated_runtime(self, runtime: str) -> bool:
        """Check if runtime is deprecated"""
        deprecated = [
            'python2.7', 'python3.6', 'nodejs8.10', 'nodejs10.x',
            'nodejs12.x', 'dotnetcore2.1', 'ruby2.5'
        ]
        return runtime in deprecated
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of Lambda collection"""
        if not self.functions:
            return {}
        
        total_functions = len(self.functions)
        total_cost = sum(f['costs']['monthly_cost'] for f in self.functions)
        total_invocations = sum(f['metrics']['invocations'] for f in self.functions)
        
        dead_functions = [f for f in self.functions if f['metrics']['invocations'] == 0]
        overprovisioned = [f for f in self.functions 
                          if f['performance']['memory_efficiency'] < 50]
        error_prone = [f for f in self.functions 
                      if f['performance']['error_rate'] > 5]
        
        return {
            'total_functions': total_functions,
            'total_monthly_cost': total_cost,
            'total_invocations': total_invocations,
            'dead_functions': len(dead_functions),
            'overprovisioned_functions': len(overprovisioned),
            'error_prone_functions': len(error_prone),
            'potential_savings': sum(
                sum(opt['potential_savings'] for opt in f['optimizations'])
                for f in self.functions
            ),
            'by_runtime': self._group_by_runtime(),
            'by_region': self._group_by_region(),
            'top_cost_functions': self._get_top_cost_functions(5)
        }
    
    def _group_by_runtime(self) -> Dict:
        """Group functions by runtime"""
        by_runtime = defaultdict(lambda: {'count': 0, 'cost': 0})
        for func in self.functions:
            runtime = func.get('runtime', 'unknown')
            by_runtime[runtime]['count'] += 1
            by_runtime[runtime]['cost'] += func['costs']['monthly_cost']
        return dict(by_runtime)
    
    def _group_by_region(self) -> Dict:
        """Group functions by region"""
        by_region = defaultdict(lambda: {'count': 0, 'cost': 0})
        for func in self.functions:
            region = func.get('region', 'unknown')
            by_region[region]['count'] += 1
            by_region[region]['cost'] += func['costs']['monthly_cost']
        return dict(by_region)
    
    def _get_top_cost_functions(self, limit: int = 5) -> List[Dict]:
        """Get top cost functions"""
        sorted_functions = sorted(
            self.functions,
            key=lambda x: x['costs']['monthly_cost'],
            reverse=True
        )
        
        return [
            {
                'name': f['name'],
                'region': f['region'],
                'monthly_cost': f['costs']['monthly_cost'],
                'invocations': f['metrics']['invocations'],
                'memory_size': f['memory_size']
            }
            for f in sorted_functions[:limit]
        ]