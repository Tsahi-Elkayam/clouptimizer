from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta

from ....core.base import (
    BaseCollector, BaseResource, ResourceType, ResourceStatus,
    ResourceMetadata, ResourceCost, ResourceMetrics
)


class EC2Resource(BaseResource):
    """EC2 instance resource"""
    pass


class EC2Collector(BaseCollector):
    """Collector for AWS EC2 instances"""
    
    def __init__(self, aws_client, config=None):
        super().__init__(aws_client, config)
        self.aws_client = aws_client
        self.logger = logging.getLogger(__name__)
    
    def collect(self) -> List[BaseResource]:
        """Collect EC2 instances from AWS"""
        self.resources = []
        
        regions = self.config.regions or self.aws_client.get_regions()
        
        for region in regions:
            try:
                self.logger.info(f"Collecting EC2 instances in {region}")
                instances = self._collect_region(region)
                self.resources.extend(instances)
            except Exception as e:
                self.logger.error(f"Error collecting EC2 in {region}: {str(e)}")
                self.errors.append({
                    "region": region,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
        
        # Enrich with metrics and costs
        self.enrich_with_metrics(self.resources)
        self.enrich_with_costs(self.resources)
        
        return self.resources
    
    def _collect_region(self, region: str) -> List[BaseResource]:
        """Collect EC2 instances in a specific region"""
        instances = []
        
        try:
            ec2_client = self.aws_client.get_client('ec2', region)
            
            # Paginate through all instances
            paginator = ec2_client.get_paginator('describe_instances')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for reservation in page['Reservations']:
                    for instance in reservation['Instances']:
                        resource = self._create_resource(instance, region)
                        if resource:
                            instances.append(resource)
        
        except Exception as e:
            self.logger.error(f"Failed to collect EC2 instances in {region}: {str(e)}")
        
        return instances
    
    def _create_resource(self, instance: Dict, region: str) -> BaseResource:
        """Create a resource object from EC2 instance data"""
        
        # Extract instance details
        instance_id = instance['InstanceId']
        instance_type = instance.get('InstanceType', 'unknown')
        state = instance['State']['Name']
        
        # Get name from tags
        name = instance_id
        for tag in instance.get('Tags', []):
            if tag['Key'] == 'Name':
                name = tag['Value']
                break
        
        # Map state to ResourceStatus
        status_map = {
            'running': ResourceStatus.RUNNING,
            'stopped': ResourceStatus.STOPPED,
            'terminated': ResourceStatus.DELETED,
            'stopping': ResourceStatus.STOPPED,
            'pending': ResourceStatus.RUNNING
        }
        status = status_map.get(state, ResourceStatus.UNKNOWN)
        
        # Create metadata
        metadata = ResourceMetadata(
            created_at=instance.get('LaunchTime'),
            tags={tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
            region=region,
            availability_zone=instance.get('Placement', {}).get('AvailabilityZone'),
            vpc_id=instance.get('VpcId'),
            subnet_id=instance.get('SubnetId')
        )
        
        # Create resource
        resource = EC2Resource(
            resource_id=instance_id,
            name=name,
            resource_type=ResourceType.COMPUTE,
            provider='aws',
            status=status,
            metadata=metadata,
            raw_data=instance
        )
        
        # Add instance-specific data
        resource.instance_type = instance_type
        resource.platform = instance.get('Platform', 'linux')
        
        return resource
    
    def get_resource_details(self, resource_id: str) -> BaseResource:
        """Get detailed information about a specific EC2 instance"""
        # Implementation would fetch specific instance details
        pass
    
    def get_resource_metrics(self, resource: BaseResource) -> Dict[str, Any]:
        """Get CloudWatch metrics for an EC2 instance"""
        
        try:
            cloudwatch = self.aws_client.get_client('cloudwatch', resource.metadata.region)
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            # Get CPU utilization
            cpu_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[
                    {'Name': 'InstanceId', 'Value': resource.resource_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            # Calculate average CPU
            cpu_avg = 0
            if cpu_response['Datapoints']:
                cpu_avg = sum(dp['Average'] for dp in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])
            
            # Get network metrics
            network_in = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='NetworkIn',
                Dimensions=[
                    {'Name': 'InstanceId', 'Value': resource.resource_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum']
            )
            
            network_out = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='NetworkOut',
                Dimensions=[
                    {'Name': 'InstanceId', 'Value': resource.resource_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum']
            )
            
            return {
                'cpu_utilization': cpu_avg,
                'network_in': sum(dp['Sum'] for dp in network_in.get('Datapoints', [])),
                'network_out': sum(dp['Sum'] for dp in network_out.get('Datapoints', []))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics for {resource.resource_id}: {str(e)}")
            return {}
    
    def get_resource_cost(self, resource: BaseResource) -> Dict[str, float]:
        """Get cost information for an EC2 instance"""
        
        # Simple cost estimation based on instance type
        # In production, this would use AWS Pricing API or Cost Explorer
        instance_costs = {
            't2.micro': 0.0116,
            't2.small': 0.023,
            't2.medium': 0.0464,
            't2.large': 0.0928,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34
        }
        
        instance_type = getattr(resource, 'instance_type', 't2.micro')
        hourly_cost = instance_costs.get(instance_type, 0.05)  # Default cost
        
        # Adjust for instance state
        if resource.status == ResourceStatus.STOPPED:
            hourly_cost = 0  # No compute cost when stopped (still pay for storage)
        
        return {
            'hourly': hourly_cost,
            'monthly': hourly_cost * 730,  # Approximate hours in a month
            'annual': hourly_cost * 8760  # Hours in a year
        }