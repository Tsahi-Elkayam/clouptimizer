"""Configuration management for Clouptimizer"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator, SecretStr
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class CloudProviderConfig(BaseModel):
    """Configuration for a cloud provider"""
    enabled: bool = True
    regions: List[str] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 100
    
    
class AWSConfig(CloudProviderConfig):
    """AWS-specific configuration"""
    profile: Optional[str] = None
    access_key_id: Optional[SecretStr] = None
    secret_access_key: Optional[SecretStr] = None
    session_token: Optional[SecretStr] = None
    assume_role_arn: Optional[str] = None
    external_id: Optional[str] = None
    mfa_serial: Optional[str] = None
    

class AzureConfig(CloudProviderConfig):
    """Azure-specific configuration"""
    tenant_id: Optional[SecretStr] = None
    client_id: Optional[SecretStr] = None
    client_secret: Optional[SecretStr] = None
    subscription_id: Optional[str] = None
    

class GCPConfig(CloudProviderConfig):
    """GCP-specific configuration"""
    project_id: Optional[str] = None
    credentials_path: Optional[Path] = None
    service_account_email: Optional[str] = None
    

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console: bool = True
    structured: bool = False
    

class SecurityConfig(BaseModel):
    """Security configuration"""
    encrypt_credentials: bool = True
    tls_verify: bool = True
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    max_request_size: int = 10485760  # 10MB
    rate_limit_per_minute: int = 60
    enable_audit_log: bool = True
    

class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = True
    ttl: int = 3600  # 1 hour
    max_size: int = 1000
    backend: str = "memory"  # memory, redis, memcached
    redis_url: Optional[str] = None
    

class NotificationConfig(BaseModel):
    """Notification configuration"""
    enabled: bool = False
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[SecretStr] = None
    smtp_tls: bool = True
    from_email: Optional[str] = None
    to_emails: List[str] = Field(default_factory=list)
    slack_webhook: Optional[SecretStr] = None
    teams_webhook: Optional[SecretStr] = None
    

class DatabaseConfig(BaseModel):
    """Database configuration for storing results"""
    enabled: bool = False
    url: Optional[SecretStr] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False
    

class ReportingConfig(BaseModel):
    """Reporting configuration"""
    output_dir: Path = Path("./reports")
    formats: List[str] = Field(default_factory=lambda: ["json", "html", "csv"])
    include_recommendations: bool = True
    include_cost_breakdown: bool = True
    include_trends: bool = True
    retention_days: int = 90
    

class OptimizationConfig(BaseModel):
    """Optimization configuration"""
    auto_apply: bool = False
    dry_run: bool = True
    approval_required: bool = True
    min_savings_threshold: float = 10.0  # Minimum savings in dollars
    confidence_threshold: float = 0.8  # Minimum confidence score
    excluded_resources: List[str] = Field(default_factory=list)
    excluded_tags: Dict[str, str] = Field(default_factory=dict)
    

class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    export_interval: int = 60
    prometheus_enabled: bool = True
    datadog_enabled: bool = False
    datadog_api_key: Optional[SecretStr] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    

class Settings(BaseSettings):
    """Main application settings"""
    app_name: str = "Clouptimizer"
    version: str = "0.1.0"
    environment: str = Field(default="development", env="CLOUPTIMIZER_ENV")
    debug: bool = Field(default=False, env="CLOUPTIMIZER_DEBUG")
    
    # Provider configurations
    aws: AWSConfig = Field(default_factory=AWSConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    
    # Feature configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Workflow settings
    max_workers: int = Field(default=10, env="CLOUPTIMIZER_MAX_WORKERS")
    batch_size: int = Field(default=100, env="CLOUPTIMIZER_BATCH_SIZE")
    scan_interval: int = Field(default=3600, env="CLOUPTIMIZER_SCAN_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "CLOUPTIMIZER_"
        case_sensitive = False
        
    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file"""
        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using defaults")
            return cls()
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(**data if data else {})
    
    @classmethod
    def from_json(cls, path: Path) -> "Settings":
        """Load settings from JSON file"""
        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using defaults")
            return cls()
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save settings to YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.dict(exclude_unset=True), f, default_flow_style=False)
    
    def to_json(self, path: Path) -> None:
        """Save settings to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.dict(exclude_unset=True), f, indent=2)
    
    def validate_providers(self) -> bool:
        """Validate that at least one provider is configured"""
        return any([
            self.aws.enabled and (self.aws.profile or self.aws.access_key_id),
            self.azure.enabled and self.azure.tenant_id,
            self.gcp.enabled and (self.gcp.project_id or self.gcp.credentials_path)
        ])
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers"""
        providers = []
        if self.aws.enabled:
            providers.append("aws")
        if self.azure.enabled:
            providers.append("azure")
        if self.gcp.enabled:
            providers.append("gcp")
        return providers


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global settings
    if settings is None:
        # Try to load from default locations
        config_paths = [
            Path.home() / ".clouptimizer" / "config.yaml",
            Path.home() / ".clouptimizer" / "config.json",
            Path("./config.yaml"),
            Path("./config.json"),
        ]
        
        for path in config_paths:
            if path.exists():
                if path.suffix == ".yaml":
                    settings = Settings.from_yaml(path)
                else:
                    settings = Settings.from_json(path)
                logger.info(f"Loaded configuration from {path}")
                break
        else:
            settings = Settings()
            logger.info("Using default configuration")
    
    return settings


def reload_settings(path: Optional[Path] = None) -> Settings:
    """Reload settings from file"""
    global settings
    
    if path:
        if path.suffix == ".yaml":
            settings = Settings.from_yaml(path)
        else:
            settings = Settings.from_json(path)
    else:
        settings = None
        settings = get_settings()
    
    return settings