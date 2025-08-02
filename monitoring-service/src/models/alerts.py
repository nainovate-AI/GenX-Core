# monitoring-service/src/models/alerts.py
"""Alert models for monitoring service"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, HttpUrl

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertState(str, Enum):
    """Alert states"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"

class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    INHIBITED = "inhibited"

class NotificationChannel(str, Enum):
    """Notification channel types"""
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    TEAMS = "teams"

class AlertLabel(BaseModel):
    """Alert label key-value pair"""
    key: str = Field(..., description="Label key")
    value: str = Field(..., description="Label value")

class AlertAnnotation(BaseModel):
    """Alert annotation for additional metadata"""
    key: str = Field(..., description="Annotation key")
    value: str = Field(..., description="Annotation value")

class AlertRule(BaseModel):
    """Alert rule definition"""
    name: str = Field(..., description="Alert rule name")
    query: str = Field(..., description="PromQL query for alert condition")
    duration: str = Field("5m", description="Duration before alert fires")
    severity: AlertSeverity = Field(AlertSeverity.WARNING, description="Alert severity")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    enabled: bool = Field(True, description="Whether rule is enabled")
    
    @validator("duration")
    def validate_duration(cls, v):
        """Validate duration format"""
        import re
        if not re.match(r'^\d+[smhd]$', v):
            raise ValueError("Duration must be in format like 5m, 1h, 30s, 1d")
        return v

class Alert(BaseModel):
    """Alert instance"""
    fingerprint: str = Field(..., description="Unique alert identifier")
    name: str = Field(..., description="Alert name")
    state: AlertState = Field(..., description="Current alert state")
    severity: AlertSeverity = Field(..., description="Alert severity")
    labels: Dict[str, str] = Field(..., description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    starts_at: datetime = Field(..., description="When alert started")
    ends_at: Optional[datetime] = Field(None, description="When alert ended")
    generator_url: Optional[str] = Field(None, description="URL to alert source")
    status: AlertStatus = Field(AlertStatus.ACTIVE, description="Alert status")
    receivers: List[str] = Field(default_factory=list, description="Alert receivers")
    
    @validator("ends_at")
    def validate_ends_at(cls, v, values):
        """Ensure ends_at is after starts_at"""
        if v and "starts_at" in values and v <= values["starts_at"]:
            raise ValueError("ends_at must be after starts_at")
        return v

class AlertGroup(BaseModel):
    """Group of related alerts"""
    group_key: str = Field(..., description="Group identifier")
    group_labels: Dict[str, str] = Field(..., description="Labels that define the group")
    alerts: List[Alert] = Field(..., description="Alerts in this group")
    total: int = Field(..., description="Total number of alerts")
    firing: int = Field(..., description="Number of firing alerts")
    resolved: int = Field(..., description="Number of resolved alerts")
    suppressed: int = Field(..., description="Number of suppressed alerts")

class AlertmanagerStatus(BaseModel):
    """AlertManager status information"""
    cluster_status: str = Field(..., description="Cluster status")
    version_info: Dict[str, str] = Field(..., description="Version information")
    uptime: str = Field(..., description="Uptime duration")
    config: Dict[str, Any] = Field(..., description="Configuration summary")

class Receiver(BaseModel):
    """Alert receiver configuration"""
    name: str = Field(..., description="Receiver name")
    webhook_configs: Optional[List["WebhookConfig"]] = Field(None, description="Webhook configurations")
    email_configs: Optional[List["EmailConfig"]] = Field(None, description="Email configurations")
    slack_configs: Optional[List["SlackConfig"]] = Field(None, description="Slack configurations")
    pagerduty_configs: Optional[List["PagerDutyConfig"]] = Field(None, description="PagerDuty configurations")

class WebhookConfig(BaseModel):
    """Webhook notification configuration"""
    url: HttpUrl = Field(..., description="Webhook URL")
    send_resolved: bool = Field(True, description="Send resolved alerts")
    max_alerts: int = Field(100, description="Maximum alerts per webhook")
    http_config: Optional[Dict[str, Any]] = Field(None, description="HTTP client configuration")

class EmailConfig(BaseModel):
    """Email notification configuration"""
    to: List[str] = Field(..., description="Email recipients")
    from_address: str = Field(..., description="Sender email address", alias="from")
    subject: Optional[str] = Field(None, description="Email subject template")
    html: Optional[str] = Field(None, description="HTML body template")
    text: Optional[str] = Field(None, description="Text body template")
    headers: Optional[Dict[str, str]] = Field(None, description="Email headers")
    send_resolved: bool = Field(True, description="Send resolved alerts")

class SlackConfig(BaseModel):
    """Slack notification configuration"""
    api_url: HttpUrl = Field(..., description="Slack webhook URL")
    channel: str = Field(..., description="Slack channel")
    username: Optional[str] = Field(None, description="Bot username")
    icon_emoji: Optional[str] = Field(None, description="Bot emoji icon")
    icon_url: Optional[HttpUrl] = Field(None, description="Bot icon URL")
    title: Optional[str] = Field(None, description="Message title template")
    text: Optional[str] = Field(None, description="Message text template")
    send_resolved: bool = Field(True, description="Send resolved alerts")
    color: Optional[str] = Field(None, description="Message color")

class PagerDutyConfig(BaseModel):
    """PagerDuty notification configuration"""
    routing_key: str = Field(..., description="PagerDuty routing key")
    service_key: Optional[str] = Field(None, description="PagerDuty service key")
    url: Optional[HttpUrl] = Field(None, description="PagerDuty API URL")
    client: Optional[str] = Field(None, description="Client name")
    client_url: Optional[HttpUrl] = Field(None, description="Client URL")
    description: Optional[str] = Field(None, description="Incident description template")
    severity: Optional[str] = Field(None, description="Incident severity")
    send_resolved: bool = Field(True, description="Send resolved alerts")

class Route(BaseModel):
    """Alert routing configuration"""
    receiver: str = Field(..., description="Receiver name")
    group_by: Optional[List[str]] = Field(None, description="Labels to group by")
    group_wait: Optional[str] = Field(None, description="Wait time before sending group")
    group_interval: Optional[str] = Field(None, description="Interval between group notifications")
    repeat_interval: Optional[str] = Field(None, description="Interval to resend alerts")
    matchers: Optional[List[str]] = Field(None, description="Label matchers")
    match: Optional[Dict[str, str]] = Field(None, description="Exact label matches")
    match_re: Optional[Dict[str, str]] = Field(None, description="Regex label matches")
    continue_: bool = Field(False, description="Continue to next route", alias="continue")
    routes: Optional[List["Route"]] = Field(None, description="Child routes")

class InhibitRule(BaseModel):
    """Alert inhibition rule"""
    source_matchers: List[str] = Field(..., description="Source alert matchers")
    target_matchers: List[str] = Field(..., description="Target alert matchers")
    equal: List[str] = Field(..., description="Labels that must be equal")

class AlertmanagerConfig(BaseModel):
    """AlertManager configuration"""
    global_config: Optional[Dict[str, Any]] = Field(None, description="Global configuration", alias="global")
    route: Route = Field(..., description="Root routing tree")
    receivers: List[Receiver] = Field(..., description="Receiver configurations")
    inhibit_rules: Optional[List[InhibitRule]] = Field(None, description="Inhibition rules")
    templates: Optional[List[str]] = Field(None, description="Template files")

class Silence(BaseModel):
    """Alert silence definition"""
    id: Optional[str] = Field(None, description="Silence ID")
    matchers: List[Dict[str, str]] = Field(..., description="Label matchers")
    starts_at: datetime = Field(..., description="Silence start time")
    ends_at: datetime = Field(..., description="Silence end time")
    created_by: str = Field(..., description="Creator username")
    comment: str = Field(..., description="Silence reason")
    status: Optional[Dict[str, Any]] = Field(None, description="Silence status")
    
    @validator("ends_at")
    def validate_ends_at(cls, v, values):
        """Ensure ends_at is after starts_at"""
        if "starts_at" in values and v <= values["starts_at"]:
            raise ValueError("ends_at must be after starts_at")
        return v

class AlertMetrics(BaseModel):
    """Alert-related metrics"""
    alerts_total: int = Field(..., description="Total number of alerts")
    alerts_firing: int = Field(..., description="Number of firing alerts")
    alerts_pending: int = Field(..., description="Number of pending alerts")
    alerts_resolved: int = Field(..., description="Number of resolved alerts")
    alerts_suppressed: int = Field(..., description="Number of suppressed alerts")
    alerts_inhibited: int = Field(..., description="Number of inhibited alerts")
    notifications_sent: int = Field(..., description="Total notifications sent")
    notifications_failed: int = Field(..., description="Failed notifications")
    silences_active: int = Field(..., description="Active silences")
    
class AlertHistoryEntry(BaseModel):
    """Alert history entry"""
    timestamp: datetime = Field(..., description="Event timestamp")
    alert_fingerprint: str = Field(..., description="Alert fingerprint")
    event_type: str = Field(..., description="Event type (fired, resolved, etc)")
    old_state: Optional[AlertState] = Field(None, description="Previous state")
    new_state: AlertState = Field(..., description="New state")
    labels: Dict[str, str] = Field(..., description="Alert labels at this time")
    annotations: Optional[Dict[str, str]] = Field(None, description="Alert annotations")
    notified_receivers: Optional[List[str]] = Field(None, description="Receivers notified")

class NotificationStatus(BaseModel):
    """Notification delivery status"""
    timestamp: datetime = Field(..., description="Notification timestamp")
    alert_fingerprint: str = Field(..., description="Alert fingerprint")
    receiver: str = Field(..., description="Receiver name")
    integration: str = Field(..., description="Integration type")
    success: bool = Field(..., description="Delivery success")
    error: Optional[str] = Field(None, description="Error message if failed")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")

# Update forward references
Route.model_rebuild()
Receiver.model_rebuild()

class AlertQueryParams(BaseModel):
    """Parameters for querying alerts"""
    state: Optional[AlertState] = Field(None, description="Filter by state")
    severity: Optional[AlertSeverity] = Field(None, description="Filter by severity")
    labels: Optional[Dict[str, str]] = Field(None, description="Filter by labels")
    receiver: Optional[str] = Field(None, description="Filter by receiver")
    silenced: Optional[bool] = Field(None, description="Include/exclude silenced")
    inhibited: Optional[bool] = Field(None, description="Include/exclude inhibited")
    active: Optional[bool] = Field(True, description="Only active alerts")
    unprocessed: Optional[bool] = Field(None, description="Only unprocessed alerts")

class CreateSilenceRequest(BaseModel):
    """Request to create a silence"""
    matchers: List[Dict[str, str]] = Field(..., description="Label matchers", min_items=1)
    starts_at: Optional[datetime] = Field(None, description="Start time (default: now)")
    ends_at: Optional[datetime] = Field(None, description="End time (required)")
    comment: str = Field(..., description="Reason for silence", min_length=1)
    created_by: str = Field(..., description="Creator username", min_length=1)
    
    @validator("starts_at", pre=True, always=True)
    def set_starts_at(cls, v):
        """Set default start time to now"""
        return v or datetime.utcnow()
    
    @validator("ends_at")
    def validate_ends_at(cls, v, values):
        """Validate end time"""
        if not v:
            raise ValueError("ends_at is required")
        if "starts_at" in values and v <= values["starts_at"]:
            raise ValueError("ends_at must be after starts_at")
        return v

class AlertWebhookPayload(BaseModel):
    """Webhook payload for alert notifications"""
    version: str = Field("4", description="Payload version")
    group_key: str = Field(..., description="Group key")
    truncated_alerts: int = Field(0, description="Number of truncated alerts")
    status: AlertState = Field(..., description="Alert status")
    receiver: str = Field(..., description="Receiver name")
    group_labels: Dict[str, str] = Field(..., description="Group labels")
    common_labels: Dict[str, str] = Field(..., description="Common labels")
    common_annotations: Dict[str, str] = Field(..., description="Common annotations")
    external_url: str = Field(..., description="AlertManager external URL")
    alerts: List[Alert] = Field(..., description="Alerts in this notification")