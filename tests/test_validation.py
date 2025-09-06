"""Tests for validation module"""

import pytest
from datetime import date, datetime

from src.core.validation import (
    Validator, ScanRequest, OptimizationRequest, ReportRequest,
    CloudProvider, ResourceType, OptimizationType
)
from src.core.exceptions import ValidationError


class TestValidator:
    """Test Validator class"""
    
    def test_validate_email(self):
        """Test email validation"""
        assert Validator.validate_email("test@example.com") == "test@example.com"
        assert Validator.validate_email("TEST@EXAMPLE.COM") == "test@example.com"
        
        with pytest.raises(ValidationError):
            Validator.validate_email("invalid-email")
        
        with pytest.raises(ValidationError):
            Validator.validate_email("@example.com")
    
    def test_validate_url(self):
        """Test URL validation"""
        assert Validator.validate_url("https://example.com") == "https://example.com"
        assert Validator.validate_url("http://example.com/path") == "http://example.com/path"
        
        with pytest.raises(ValidationError):
            Validator.validate_url("not-a-url")
        
        with pytest.raises(ValidationError):
            Validator.validate_url("ftp://example.com")
    
    def test_validate_ip_address(self):
        """Test IP address validation"""
        assert Validator.validate_ip_address("192.168.1.1") == "192.168.1.1"
        assert Validator.validate_ip_address("::1") == "::1"
        
        with pytest.raises(ValidationError):
            Validator.validate_ip_address("256.256.256.256")
        
        with pytest.raises(ValidationError):
            Validator.validate_ip_address("not-an-ip")
    
    def test_validate_port(self):
        """Test port validation"""
        assert Validator.validate_port(80) == 80
        assert Validator.validate_port("8080") == 8080
        assert Validator.validate_port(65535) == 65535
        
        with pytest.raises(ValidationError):
            Validator.validate_port(0)
        
        with pytest.raises(ValidationError):
            Validator.validate_port(65536)
        
        with pytest.raises(ValidationError):
            Validator.validate_port("not-a-port")
    
    def test_validate_aws_account_id(self):
        """Test AWS account ID validation"""
        assert Validator.validate_aws_account_id("123456789012") == "123456789012"
        
        with pytest.raises(ValidationError):
            Validator.validate_aws_account_id("12345678901")  # Too short
        
        with pytest.raises(ValidationError):
            Validator.validate_aws_account_id("1234567890123")  # Too long
        
        with pytest.raises(ValidationError):
            Validator.validate_aws_account_id("12345678901a")  # Contains letter
    
    def test_validate_aws_region(self):
        """Test AWS region validation"""
        assert Validator.validate_aws_region("us-east-1") == "us-east-1"
        assert Validator.validate_aws_region("eu-west-1") == "eu-west-1"
        
        with pytest.raises(ValidationError):
            Validator.validate_aws_region("invalid-region")
        
        with pytest.raises(ValidationError):
            Validator.validate_aws_region("us-east")
    
    def test_validate_azure_subscription(self):
        """Test Azure subscription ID validation"""
        valid_id = "12345678-1234-1234-1234-123456789012"
        assert Validator.validate_azure_subscription(valid_id) == valid_id
        
        with pytest.raises(ValidationError):
            Validator.validate_azure_subscription("invalid-id")
        
        with pytest.raises(ValidationError):
            Validator.validate_azure_subscription("12345678-1234-1234-1234")
    
    def test_validate_date_range(self):
        """Test date range validation"""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        
        result = Validator.validate_date_range(start, end)
        assert result == (start, end)
        
        # Test with string dates
        result = Validator.validate_date_range("2024-01-01", "2024-12-31")
        assert result == (start, end)
        
        with pytest.raises(ValidationError):
            Validator.validate_date_range(end, start)  # End before start
    
    def test_sanitize_string(self):
        """Test string sanitization"""
        assert Validator.sanitize_string("  test  ") == "test"
        assert Validator.sanitize_string("test-string_123") == "test-string_123"
        assert Validator.sanitize_string("test@#$%") == "test"
        
        # Test max length
        long_string = "a" * 300
        assert len(Validator.sanitize_string(long_string)) == 255
    
    def test_sanitize_html(self):
        """Test HTML sanitization"""
        assert Validator.sanitize_html("<p>Hello</p>") == "<p>Hello</p>"
        assert Validator.sanitize_html("<script>alert('xss')</script>") == ""
        assert Validator.sanitize_html('<a href="javascript:alert()">link</a>') == '<a href="">link</a>'
        assert Validator.sanitize_html('<div onclick="alert()">text</div>') == '<div>text</div>'
    
    def test_validate_cloud_provider(self):
        """Test cloud provider validation"""
        assert Validator.validate_cloud_provider("aws") == "aws"
        assert Validator.validate_cloud_provider("AWS") == "aws"
        assert Validator.validate_cloud_provider("azure") == "azure"
        assert Validator.validate_cloud_provider("gcp") == "gcp"
        
        with pytest.raises(ValidationError):
            Validator.validate_cloud_provider("invalid")
    
    def test_validate_batch(self):
        """Test batch validation"""
        items = ["test@example.com", "user@domain.com"]
        result = Validator.validate_batch(items, Validator.validate_email)
        assert result == ["test@example.com", "user@domain.com"]
        
        # Test with invalid item
        items = ["test@example.com", "invalid-email"]
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_batch(items, Validator.validate_email)
        assert "Item 1" in str(exc_info.value)
        
        # Test fail_fast
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_batch(items, Validator.validate_email, fail_fast=True)
        assert "item 1" in str(exc_info.value)


class TestRequestValidators:
    """Test request validator models"""
    
    def test_scan_request(self):
        """Test ScanRequest validation"""
        request = ScanRequest(
            provider=CloudProvider.AWS,
            regions=["us-east-1"],
            services=["ec2", "s3"],
            resource_types=[ResourceType.COMPUTE, ResourceType.STORAGE]
        )
        
        assert request.provider == "aws"
        assert request.regions == ["us-east-1"]
        assert request.resource_types == ["compute", "storage"]
        
        # Test invalid region for AWS
        with pytest.raises(ValueError):
            ScanRequest(
                provider=CloudProvider.AWS,
                regions=["invalid-region"]
            )
    
    def test_optimization_request(self):
        """Test OptimizationRequest validation"""
        request = OptimizationRequest(
            optimization_types=[OptimizationType.RIGHTSIZING, OptimizationType.IDLE_RESOURCES],
            dry_run=True,
            min_savings=100.0,
            confidence_threshold=0.8
        )
        
        assert request.optimization_types == ["rightsizing", "idle_resources"]
        assert request.dry_run is True
        assert request.min_savings == 100.0
        assert request.confidence_threshold == 0.8
        
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            OptimizationRequest(
                optimization_types=[OptimizationType.RIGHTSIZING],
                confidence_threshold=1.5  # > 1
            )
    
    def test_report_request(self):
        """Test ReportRequest validation"""
        request = ReportRequest(
            report_type="cost_summary",
            format="json",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )
        
        assert request.report_type == "cost_summary"
        assert request.format == "json"
        assert request.start_date == date(2024, 1, 1)
        assert request.end_date == date(2024, 12, 31)
        
        # Test invalid format
        with pytest.raises(ValueError):
            ReportRequest(
                report_type="cost_summary",
                format="invalid",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31)
            )
        
        # Test invalid date range
        with pytest.raises(ValueError):
            ReportRequest(
                report_type="cost_summary",
                format="json",
                start_date=date(2024, 12, 31),
                end_date=date(2024, 1, 1)  # End before start
            )