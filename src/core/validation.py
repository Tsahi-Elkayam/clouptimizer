"""Input validation and sanitization utilities"""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from pathlib import Path
import json
import yaml
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum

from .exceptions import ValidationError as CustomValidationError


class CloudProvider(str, Enum):
    """Valid cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class ResourceType(str, Enum):
    """Valid resource types"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    ANALYTICS = "analytics"
    AI_ML = "ai_ml"
    SECURITY = "security"
    MONITORING = "monitoring"


class OptimizationType(str, Enum):
    """Valid optimization types"""
    RIGHTSIZING = "rightsizing"
    RESERVED_INSTANCES = "reserved_instances"
    SAVINGS_PLANS = "savings_plans"
    SPOT_INSTANCES = "spot_instances"
    IDLE_RESOURCES = "idle_resources"
    ORPHANED_RESOURCES = "orphaned_resources"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    LICENSE_OPTIMIZATION = "license_optimization"


class Validator:
    """Central validation utility"""
    
    # Regex patterns for validation
    PATTERNS = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
        'aws_account_id': re.compile(r'^\d{12}$'),
        'aws_region': re.compile(r'^[a-z]{2}-[a-z]+-\d{1}$'),
        'aws_arn': re.compile(r'^arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d{12}:.*'),
        'azure_subscription': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        'gcp_project': re.compile(r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$'),
        'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'safe_string': re.compile(r'^[a-zA-Z0-9_.-]+$'),
    }
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email address"""
        if not cls.PATTERNS['email'].match(email):
            raise CustomValidationError(f"Invalid email address: {email}")
        return email.lower()
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate URL"""
        if not cls.PATTERNS['url'].match(url):
            raise CustomValidationError(f"Invalid URL: {url}")
        return url
    
    @classmethod
    def validate_ip_address(cls, ip: str) -> str:
        """Validate IP address (IPv4 or IPv6)"""
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            raise CustomValidationError(f"Invalid IP address: {ip}")
    
    @classmethod
    def validate_ip_network(cls, network: str) -> str:
        """Validate IP network"""
        try:
            ipaddress.ip_network(network)
            return network
        except ValueError:
            raise CustomValidationError(f"Invalid IP network: {network}")
    
    @classmethod
    def validate_port(cls, port: Union[int, str]) -> int:
        """Validate port number"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                raise CustomValidationError(f"Port must be between 1 and 65535: {port}")
            return port_num
        except (ValueError, TypeError):
            raise CustomValidationError(f"Invalid port number: {port}")
    
    @classmethod
    def validate_aws_account_id(cls, account_id: str) -> str:
        """Validate AWS account ID"""
        if not cls.PATTERNS['aws_account_id'].match(account_id):
            raise CustomValidationError(f"Invalid AWS account ID: {account_id}")
        return account_id
    
    @classmethod
    def validate_aws_region(cls, region: str) -> str:
        """Validate AWS region"""
        valid_regions = [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1',
            'eu-north-1', 'ap-south-1', 'ap-southeast-1', 'ap-southeast-2',
            'ap-northeast-1', 'ap-northeast-2', 'ap-northeast-3',
            'sa-east-1', 'ca-central-1', 'me-south-1', 'af-south-1'
        ]
        if region not in valid_regions:
            raise CustomValidationError(f"Invalid AWS region: {region}")
        return region
    
    @classmethod
    def validate_aws_arn(cls, arn: str) -> str:
        """Validate AWS ARN"""
        if not cls.PATTERNS['aws_arn'].match(arn):
            raise CustomValidationError(f"Invalid AWS ARN: {arn}")
        return arn
    
    @classmethod
    def validate_azure_subscription(cls, subscription_id: str) -> str:
        """Validate Azure subscription ID"""
        if not cls.PATTERNS['azure_subscription'].match(subscription_id):
            raise CustomValidationError(f"Invalid Azure subscription ID: {subscription_id}")
        return subscription_id
    
    @classmethod
    def validate_gcp_project(cls, project_id: str) -> str:
        """Validate GCP project ID"""
        if not cls.PATTERNS['gcp_project'].match(project_id):
            raise CustomValidationError(f"Invalid GCP project ID: {project_id}")
        return project_id
    
    @classmethod
    def validate_uuid(cls, uuid_str: str) -> str:
        """Validate UUID"""
        if not cls.PATTERNS['uuid'].match(uuid_str.lower()):
            raise CustomValidationError(f"Invalid UUID: {uuid_str}")
        return uuid_str.lower()
    
    @classmethod
    def validate_date_range(cls, start_date: Union[str, date], end_date: Union[str, date]) -> tuple:
        """Validate date range"""
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
        
        if start_date > end_date:
            raise CustomValidationError(f"Start date must be before end date")
        
        return start_date, end_date
    
    @classmethod
    def validate_json(cls, json_str: str) -> Dict[str, Any]:
        """Validate and parse JSON string"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise CustomValidationError(f"Invalid JSON: {e}")
    
    @classmethod
    def validate_yaml(cls, yaml_str: str) -> Dict[str, Any]:
        """Validate and parse YAML string"""
        try:
            return yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise CustomValidationError(f"Invalid YAML: {e}")
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path], must_exist: bool = False) -> Path:
        """Validate file path"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if must_exist and not path.exists():
            raise CustomValidationError(f"File does not exist: {path}")
        
        # Check for path traversal attempts
        try:
            path.resolve()
        except Exception:
            raise CustomValidationError(f"Invalid file path: {file_path}")
        
        return path
    
    @classmethod
    def sanitize_string(cls, input_str: str, pattern: str = 'safe_string', 
                       max_length: int = 255) -> str:
        """Sanitize string input"""
        if not input_str:
            return ""
        
        # Remove leading/trailing whitespace
        sanitized = input_str.strip()
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Check against pattern if specified
        if pattern in cls.PATTERNS:
            if not cls.PATTERNS[pattern].match(sanitized):
                # Remove invalid characters
                sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '', sanitized)
        
        return sanitized
    
    @classmethod
    def sanitize_html(cls, html_str: str) -> str:
        """Sanitize HTML to prevent XSS"""
        # Remove script tags and other dangerous elements
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'form']
        sanitized = html_str
        
        for tag in dangerous_tags:
            sanitized = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(f'<{tag}[^>]*>', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        
        # Remove javascript: protocol
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @classmethod
    def validate_cloud_provider(cls, provider: str) -> str:
        """Validate cloud provider"""
        try:
            return CloudProvider(provider.lower()).value
        except ValueError:
            raise CustomValidationError(f"Invalid cloud provider: {provider}")
    
    @classmethod
    def validate_resource_type(cls, resource_type: str) -> str:
        """Validate resource type"""
        try:
            return ResourceType(resource_type.lower()).value
        except ValueError:
            raise CustomValidationError(f"Invalid resource type: {resource_type}")
    
    @classmethod
    def validate_batch(cls, items: List[Any], validator: Callable[[Any], Any], 
                      fail_fast: bool = False) -> List[Any]:
        """Validate a batch of items"""
        validated = []
        errors = []
        
        for i, item in enumerate(items):
            try:
                validated.append(validator(item))
            except Exception as e:
                if fail_fast:
                    raise CustomValidationError(f"Validation failed at item {i}: {e}")
                errors.append(f"Item {i}: {e}")
        
        if errors and not fail_fast:
            raise CustomValidationError(f"Batch validation failed: {'; '.join(errors)}")
        
        return validated


class RequestValidator(BaseModel):
    """Base model for request validation"""
    
    class Config:
        str_strip_whitespace = True
        use_enum_values = True
        validate_assignment = True
        

class ScanRequest(RequestValidator):
    """Validate scan request"""
    provider: CloudProvider
    regions: List[str] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    resource_types: List[ResourceType] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    exclude_tags: Dict[str, str] = Field(default_factory=dict)
    max_age_days: int = Field(default=30, ge=1, le=365)
    
    @validator('regions', each_item=True)
    def validate_region(cls, region: str, values: dict) -> str:
        provider = values.get('provider')
        if provider == CloudProvider.AWS:
            return Validator.validate_aws_region(region)
        return region


class OptimizationRequest(RequestValidator):
    """Validate optimization request"""
    optimization_types: List[OptimizationType]
    dry_run: bool = True
    auto_apply: bool = False
    min_savings: float = Field(default=0, ge=0)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    exclude_resources: List[str] = Field(default_factory=list)
    

class ReportRequest(RequestValidator):
    """Validate report request"""
    report_type: str = Field(regex='^[a-z_]+$')
    format: str = Field(default='json', regex='^(json|html|csv|pdf)$')
    start_date: date
    end_date: date
    include_recommendations: bool = True
    include_trends: bool = True
    group_by: List[str] = Field(default_factory=list)
    
    @validator('end_date')
    def validate_dates(cls, end_date: date, values: dict) -> date:
        start_date = values.get('start_date')
        if start_date and start_date > end_date:
            raise ValueError("Start date must be before end date")
        return end_date