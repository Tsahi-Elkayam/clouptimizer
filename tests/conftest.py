"""Pytest configuration and fixtures"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml
import json

from src.core.config import Settings, get_settings
from src.core.monitoring import MetricsCollector, HealthChecker
from src.core.security import TokenManager, EncryptionManager, APIKeyManager
from src.core.logging import setup_logging


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings"""
    return Settings(
        environment="test",
        debug=True,
        aws={
            "enabled": True,
            "regions": ["us-east-1"],
            "services": ["ec2", "s3"]
        },
        azure={
            "enabled": False
        },
        gcp={
            "enabled": False
        },
        logging={
            "level": "DEBUG",
            "structured": False,
            "console": False
        },
        security={
            "rate_limit_per_minute": 100
        },
        cache={
            "enabled": False
        },
        database={
            "enabled": False
        }
    )


@pytest.fixture
def temp_config_file():
    """Create temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            "app_name": "Clouptimizer Test",
            "environment": "test",
            "aws": {
                "enabled": True,
                "regions": ["us-east-1"]
            }
        }
        yaml.dump(config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def metrics_collector():
    """Create metrics collector"""
    collector = MetricsCollector()
    collector.start()
    yield collector
    collector.stop()


@pytest.fixture
def health_checker():
    """Create health checker"""
    return HealthChecker()


@pytest.fixture
def token_manager():
    """Create token manager"""
    return TokenManager(secret_key="test_secret_key")


@pytest.fixture
def encryption_manager():
    """Create encryption manager"""
    return EncryptionManager()


@pytest.fixture
def api_key_manager():
    """Create API key manager"""
    return APIKeyManager()


@pytest.fixture
def mock_aws_client():
    """Mock AWS client"""
    with patch('boto3.Session') as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock STS responses
        mock_client.get_caller_identity.return_value = {
            'Account': '123456789012',
            'Arn': 'arn:aws:iam::123456789012:user/test',
            'UserId': 'AIDACKCEVSQ6C2EXAMPLE'
        }
        
        # Mock EC2 responses
        mock_client.describe_regions.return_value = {
            'Regions': [
                {'RegionName': 'us-east-1'},
                {'RegionName': 'us-west-2'}
            ]
        }
        
        mock_client.describe_instances.return_value = {
            'Reservations': [
                {
                    'Instances': [
                        {
                            'InstanceId': 'i-1234567890abcdef0',
                            'InstanceType': 't2.micro',
                            'State': {'Name': 'running'},
                            'LaunchTime': '2024-01-01T00:00:00Z'
                        }
                    ]
                }
            ]
        }
        
        yield mock_client


@pytest.fixture
def mock_azure_client():
    """Mock Azure client"""
    mock_client = MagicMock()
    # Add Azure-specific mocks
    yield mock_client


@pytest.fixture
def mock_gcp_client():
    """Mock GCP client"""
    mock_client = MagicMock()
    # Add GCP-specific mocks
    yield mock_client


@pytest.fixture
def sample_resources():
    """Sample cloud resources for testing"""
    return [
        {
            "id": "i-1234567890abcdef0",
            "type": "ec2_instance",
            "provider": "aws",
            "region": "us-east-1",
            "name": "test-instance",
            "tags": {"Environment": "test"},
            "cost": 10.0,
            "metrics": {
                "cpu_utilization": 5.0,
                "memory_utilization": 20.0
            }
        },
        {
            "id": "vol-0987654321fedcba",
            "type": "ebs_volume",
            "provider": "aws",
            "region": "us-east-1",
            "name": "test-volume",
            "tags": {"Environment": "test"},
            "cost": 5.0,
            "metrics": {
                "iops": 100,
                "throughput": 125
            }
        }
    ]


@pytest.fixture
def sample_optimization_recommendations():
    """Sample optimization recommendations"""
    return [
        {
            "resource_id": "i-1234567890abcdef0",
            "type": "rightsizing",
            "description": "Downsize instance from t2.micro to t2.nano",
            "estimated_savings": 5.0,
            "confidence": 0.85,
            "risk": "low",
            "implementation": {
                "action": "modify_instance_type",
                "parameters": {
                    "instance_id": "i-1234567890abcdef0",
                    "new_type": "t2.nano"
                }
            }
        },
        {
            "resource_id": "vol-0987654321fedcba",
            "type": "idle_resource",
            "description": "Delete unattached EBS volume",
            "estimated_savings": 5.0,
            "confidence": 0.95,
            "risk": "medium",
            "implementation": {
                "action": "delete_volume",
                "parameters": {
                    "volume_id": "vol-0987654321fedcba"
                }
            }
        }
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset any global singletons
    import src.core.config as config_module
    config_module.settings = None
    yield
    config_module.settings = None


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    env_vars = {
        "CLOUPTIMIZER_ENV": "test",
        "CLOUPTIMIZER_DEBUG": "true",
        "AWS_PROFILE": "test",
        "AWS_REGION": "us-east-1"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "aws: mark test as AWS-specific"
    )
    config.addinivalue_line(
        "markers", "azure: mark test as Azure-specific"
    )
    config.addinivalue_line(
        "markers", "gcp: mark test as GCP-specific"
    )