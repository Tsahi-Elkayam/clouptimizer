import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from typing import Dict, List, Any, Optional

from ...core.base import BaseCloudProvider, CloudCredentials, CloudProvider


class AWSClient(BaseCloudProvider):
    """AWS cloud provider implementation"""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.session = None
        self.clients: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def authenticate(self) -> bool:
        """Authenticate with AWS"""
        try:
            # Create session
            if self.credentials.profile:
                self.session = boto3.Session(profile_name=self.credentials.profile)
            else:
                self.session = boto3.Session()
            
            # Test authentication by calling STS
            sts_client = self.session.client('sts')
            identity = sts_client.get_caller_identity()
            
            self.logger.info(f"Authenticated as: {identity['Arn']}")
            self._authenticated = True
            return True
            
        except NoCredentialsError:
            self.logger.error("No AWS credentials found")
            self._authenticated = False
            return False
        except Exception as e:
            self.logger.error(f"AWS authentication failed: {str(e)}")
            self._authenticated = False
            return False
    
    def validate_credentials(self) -> bool:
        """Validate AWS credentials"""
        try:
            sts_client = self.get_client('sts')
            sts_client.get_caller_identity()
            return True
        except:
            return False
    
    def get_regions(self) -> List[str]:
        """Get list of available AWS regions"""
        try:
            ec2_client = self.get_client('ec2', region='us-east-1')
            response = ec2_client.describe_regions()
            return [region['RegionName'] for region in response['Regions']]
        except Exception as e:
            self.logger.error(f"Failed to get regions: {str(e)}")
            return []
    
    def get_services(self) -> List[str]:
        """Get list of available AWS services"""
        # Return common AWS services
        return [
            'ec2', 's3', 'rds', 'lambda', 'dynamodb',
            'elasticache', 'ecs', 'eks', 'elb', 'cloudfront',
            'cloudwatch', 'sns', 'sqs', 'kinesis', 'apigateway'
        ]
    
    def test_connection(self) -> bool:
        """Test connection to AWS"""
        try:
            ec2_client = self.get_client('ec2')
            ec2_client.describe_instances(MaxResults=1)
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_client(self, service: str, region: str = None):
        """Get or create a boto3 client for a service"""
        if not self.session:
            if not self.authenticate():
                raise Exception("Not authenticated")
        
        region = region or self.credentials.region or 'us-east-1'
        client_key = f"{service}_{region}"
        
        if client_key not in self.clients:
            self.clients[client_key] = self.session.client(service, region_name=region)
        
        return self.clients[client_key]
    
    def get_resource(self, service: str, region: str = None):
        """Get or create a boto3 resource for a service"""
        if not self.session:
            if not self.authenticate():
                raise Exception("Not authenticated")
        
        region = region or self.credentials.region or 'us-east-1'
        return self.session.resource(service, region_name=region)
    
    def get_account_id(self) -> str:
        """Get AWS account ID"""
        try:
            sts_client = self.get_client('sts')
            return sts_client.get_caller_identity()['Account']
        except:
            return ""
    
    def get_cost_data(self, start_date: str, end_date: str, granularity: str = 'MONTHLY') -> Dict:
        """Get cost data from AWS Cost Explorer"""
        try:
            ce_client = self.get_client('ce', region='us-east-1')
            
            response = ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity=granularity,
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            return response
        except Exception as e:
            self.logger.error(f"Failed to get cost data: {str(e)}")
            return {}