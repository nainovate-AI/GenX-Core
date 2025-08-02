# monitoring-service/src/core/config.py
"""Configuration management for monitoring service"""

from pydantic import BaseSettings, Field, validator
from typing import List, Optional, Dict
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Service configuration
    SERVICE_NAME: str = Field("monitoring-service", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field("1.0.0", env="SERVICE_VERSION")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # API configuration
    API_PREFIX: str = Field("/api/v1", env="API_PREFIX")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    CORS_ORIGINS: List[str] = Field(["*"], env="CORS_ORIGINS")
    
    # Component URLs
    PROMETHEUS_URL: str = Field("http://prometheus:9090", env="PROMETHEUS_URL")
    LOKI_URL: str = Field("http://loki:3100", env="LOKI_URL")
    JAEGER_URL: str = Field("http://jaeger:16686", env="JAEGER_URL")
    GRAFANA_URL: str = Field("http://grafana:3000", env="GRAFANA_URL")
    ALERTMANAGER_URL: str = Field("http://alertmanager:9093", env="ALERTMANAGER_URL")
    OTEL_COLLECTOR_URL: str = Field("otel-collector:4317", env="OTEL_COLLECTOR_URL")
    
    # Internal endpoints (for direct API access)
    JAEGER_QUERY_URL: str = Field("http://jaeger:16686", env="JAEGER_QUERY_URL")
    LOKI_PUSH_URL: str = Field("http://loki:3100/loki/api/v1/push", env="LOKI_PUSH_URL")
    
    # Docker configuration
    DOCKER_NETWORK: str = Field("monitoring-network", env="DOCKER_NETWORK")
    DOCKER_SOCKET: str = Field("/var/run/docker.sock", env="DOCKER_SOCKET")
    DOCKER_COMPOSE_FILE: str = Field("docker/docker-compose.yml", env="DOCKER_COMPOSE_FILE")
    
    # Storage configuration
    DATA_RETENTION_DAYS: int = Field(30, env="DATA_RETENTION_DAYS")
    PROMETHEUS_RETENTION_SIZE: str = Field("10GB", env="PROMETHEUS_RETENTION_SIZE")
    LOKI_RETENTION_HOURS: int = Field(720, env="LOKI_RETENTION_HOURS")  # 30 days
    JAEGER_MAX_TRACES: int = Field(50000, env="JAEGER_MAX_TRACES")
    
    # Performance configuration
    MAX_QUERY_TIMEOUT: int = Field(300, env="MAX_QUERY_TIMEOUT")  # 5 minutes
    DEFAULT_QUERY_TIMEOUT: int = Field(30, env="DEFAULT_QUERY_TIMEOUT")
    MAX_QUERY_SAMPLES: int = Field(50000000, env="MAX_QUERY_SAMPLES")  # 50M samples
    
    # Grafana configuration
    GRAFANA_ADMIN_USER: str = Field("admin", env="GRAFANA_ADMIN_USER")
    GRAFANA_ADMIN_PASSWORD: str = Field("admin", env="GRAFANA_ADMIN_PASSWORD")
    GRAFANA_ORG_ID: int = Field(1, env="GRAFANA_ORG_ID")
    
    # OpenTelemetry configuration
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field("otel-collector:4317", env="OTEL_EXPORTER_OTLP_ENDPOINT")
    OTEL_EXPORTER_OTLP_INSECURE: bool = Field(True, env="OTEL_EXPORTER_OTLP_INSECURE")
    OTEL_METRICS_EXPORTER: str = Field("otlp", env="OTEL_METRICS_EXPORTER")
    OTEL_TRACES_EXPORTER: str = Field("otlp", env="OTEL_TRACES_EXPORTER")
    OTEL_LOGS_EXPORTER: str = Field("otlp", env="OTEL_LOGS_EXPORTER")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW_SECONDS: int = Field(60, env="RATE_LIMIT_WINDOW_SECONDS")
    
    # Health check configuration
    HEALTH_CHECK_INTERVAL: int = Field(30, env="HEALTH_CHECK_INTERVAL")  # seconds
    HEALTH_CHECK_TIMEOUT: int = Field(10, env="HEALTH_CHECK_TIMEOUT")
    
    # Alert configuration
    ALERT_WEBHOOK_URL: Optional[str] = Field(None, env="ALERT_WEBHOOK_URL")
    ALERT_EMAIL_ENABLED: bool = Field(False, env="ALERT_EMAIL_ENABLED")
    ALERT_EMAIL_SMTP_HOST: Optional[str] = Field(None, env="ALERT_EMAIL_SMTP_HOST")
    ALERT_EMAIL_FROM: Optional[str] = Field(None, env="ALERT_EMAIL_FROM")
    ALERT_EMAIL_TO: Optional[List[str]] = Field(None, env="ALERT_EMAIL_TO")
    
    # Security configuration (for future use)
    JWT_SECRET_KEY: Optional[str] = Field(None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(24, env="JWT_EXPIRATION_HOURS")
    API_KEY_ENABLED: bool = Field(False, env="API_KEY_ENABLED")
    
    # Feature flags
    ENABLE_METRICS_ENDPOINT: bool = Field(True, env="ENABLE_METRICS_ENDPOINT")
    ENABLE_PROFILING: bool = Field(False, env="ENABLE_PROFILING")
    ENABLE_TRACE_SAMPLING: bool = Field(True, env="ENABLE_TRACE_SAMPLING")
    TRACE_SAMPLING_RATE: float = Field(0.1, env="TRACE_SAMPLING_RATE")  # 10%
    
    # Paths
    CONFIG_PATH: Path = Field(Path("/app/configs"), env="CONFIG_PATH")
    DASHBOARDS_PATH: Path = Field(Path("/app/configs/grafana/dashboards"), env="DASHBOARDS_PATH")
    ALERTS_PATH: Path = Field(Path("/app/configs/prometheus/alerts"), env="ALERTS_PATH")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALERT_EMAIL_TO", pre=True)
    def parse_email_list(cls, v):
        """Parse email list from comma-separated string"""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()
    
    @validator("TRACE_SAMPLING_RATE")
    def validate_sampling_rate(cls, v):
        """Validate trace sampling rate"""
        if not 0 <= v <= 1:
            raise ValueError("Trace sampling rate must be between 0 and 1")
        return v
    
    def get_component_url(self, component: str) -> str:
        """Get URL for a monitoring component"""
        url_map = {
            "prometheus": self.PROMETHEUS_URL,
            "loki": self.LOKI_URL,
            "jaeger": self.JAEGER_URL,
            "grafana": self.GRAFANA_URL,
            "alertmanager": self.ALERTMANAGER_URL,
            "otel-collector": f"http://{self.OTEL_COLLECTOR_URL}"
        }
        return url_map.get(component.lower(), "")
    
    def get_retention_settings(self) -> Dict[str, any]:
        """Get retention settings for all components"""
        return {
            "prometheus": {
                "days": self.DATA_RETENTION_DAYS,
                "size": self.PROMETHEUS_RETENTION_SIZE
            },
            "loki": {
                "hours": self.LOKI_RETENTION_HOURS
            },
            "jaeger": {
                "max_traces": self.JAEGER_MAX_TRACES
            }
        }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Allow extra fields for forward compatibility
        extra = "allow"

# Create global settings instance
settings = Settings()

# Export commonly used settings
SERVICE_NAME = settings.SERVICE_NAME
ENVIRONMENT = settings.ENVIRONMENT
DEBUG = settings.DEBUG

# Validate critical settings on startup
def validate_settings():
    """Validate critical settings on startup"""
    if settings.is_production:
        # Production validations
        if settings.DEBUG:
            raise ValueError("DEBUG must be False in production")
        
        if settings.GRAFANA_ADMIN_PASSWORD == "admin":
            raise ValueError("Change default Grafana admin password in production")
        
        if settings.API_KEY_ENABLED and not settings.JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY must be set when API_KEY_ENABLED is True")
    
    # Check paths exist
    if not settings.CONFIG_PATH.exists():
        raise ValueError(f"Config path does not exist: {settings.CONFIG_PATH}")
    
    return True

# Run validation if not in test mode
if os.getenv("TESTING", "false").lower() != "true":
    validate_settings()