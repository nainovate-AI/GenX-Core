# monitoring-service/src/models/__init__.py
"""Models module for monitoring service"""

from .alerts import (
    # Enums
    AlertSeverity,
    AlertState,
    AlertStatus,
    NotificationChannel,
    
    # Core Models
    AlertLabel,
    AlertAnnotation,
    AlertRule,
    Alert,
    AlertGroup,
    AlertmanagerStatus,
    
    # Notification Models
    Receiver,
    WebhookConfig,
    EmailConfig,
    SlackConfig,
    PagerDutyConfig,
    
    # Configuration Models
    Route,
    InhibitRule,
    AlertmanagerConfig,
    Silence,
    
    # Metric and History Models
    AlertMetrics,
    AlertHistoryEntry,
    NotificationStatus,
    
    # Request/Response Models
    AlertQueryParams,
    CreateSilenceRequest,
    AlertWebhookPayload
)

from .logs import (
    # Enums
    LogLevel,
    LogDirection,
    
    # Core Models
    LogEntry,
    LogStream,
    LogQuery,
    LogQueryResponse,
    LogStreamResponse,
    
    # Label Models
    LogLabelsResponse,
    LogLabelValuesResponse,
    LogSeriesResponse,
    
    # Metric Models
    LogMetricQuery,
    LogMetricResponse,
    
    # Statistics Models
    LogStats,
    LogStreamStats,
    
    # Request/Response Models
    LogTailRequest,
    LogContextRequest,
    LogContextResponse,
    LogFilter,
    LogExportRequest,
    LogIngestionRequest,
    
    # Advanced Models
    LogParseRule,
    LogAggregation,
    LogPattern,
    LogAnomaly,
    LogQueryBuilder,
    LogRetentionPolicy
)

from .metrics import (
    # Enums
    MetricType,
    ResultType,
    TargetHealth,
    RuleType,
    RuleHealth,
    
    # Core Models
    MetricMetadata,
    Label,
    Sample,
    InstantVector,
    RangeVector,
    ScalarResult,
    StringResult,
    
    # Query Models
    MetricQuery,
    MetricRangeQuery,
    MetricResponse,
    
    # Target Models
    Target,
    TargetsResponse,
    
    # Rule Models
    Rule,
    RuleGroup,
    RulesResponse,
    
    # Alert Models
    Alert as PrometheusAlert,
    AlertsResponse,
    
    # System Models
    TSDBStatus,
    SeriesMetadata,
    QueryExemplar,
    BuildInfo,
    RuntimeInfo,
    
    # Helper Models
    QueryBuilder,
    MetricAggregation,
    CommonQueries,
    
    # Request/Response Models
    SnapshotRequest,
    SnapshotResponse,
    DeleteSeriesRequest,
    MetricStatistics
)

from .traces import (
    # Enums
    SpanKind,
    SpanStatus,
    ValueType,
    ReferenceType,
    
    # Core Models
    ProcessTag,
    Process,
    SpanLog,
    SpanReference,
    SpanTag,
    Span,
    Trace,
    
    # Search Models
    TraceSearchParams,
    TraceSearchResponse,
    
    # Service Models
    Service,
    Operation,
    ServiceDependency,
    ServiceDependencies,
    
    # Metric Models
    TraceTimeline,
    TraceMetrics,
    ServiceMetrics,
    TraceComparison,
    LatencyPercentiles,
    
    # Visualization Models
    FlamegraphNode,
    Flamegraph,
    ServiceNode,
    ServiceEdge,
    ServiceMap,
    
    # Statistics Models
    SpanKindStats,
    TraceStatistics,
    
    # Configuration Models
    SamplingConfig,
    StorageInfo,
    JaegerConfig,
    
    # Debug Models
    TraceDebugInfo,
    TraceQueryBuilder,
    SpanContext
)

__all__ = [
    # Alert Models
    "AlertSeverity",
    "AlertState",
    "AlertStatus",
    "NotificationChannel",
    "AlertLabel",
    "AlertAnnotation",
    "AlertRule",
    "Alert",
    "AlertGroup",
    "AlertmanagerStatus",
    "Receiver",
    "WebhookConfig",
    "EmailConfig",
    "SlackConfig",
    "PagerDutyConfig",
    "Route",
    "InhibitRule",
    "AlertmanagerConfig",
    "Silence",
    "AlertMetrics",
    "AlertHistoryEntry",
    "NotificationStatus",
    "AlertQueryParams",
    "CreateSilenceRequest",
    "AlertWebhookPayload",
    
    # Log Models
    "LogLevel",
    "LogDirection",
    "LogEntry",
    "LogStream",
    "LogQuery",
    "LogQueryResponse",
    "LogStreamResponse",
    "LogLabelsResponse",
    "LogLabelValuesResponse",
    "LogSeriesResponse",
    "LogMetricQuery",
    "LogMetricResponse",
    "LogStats",
    "LogStreamStats",
    "LogTailRequest",
    "LogContextRequest",
    "LogContextResponse",
    "LogFilter",
    "LogExportRequest",
    "LogIngestionRequest",
    "LogParseRule",
    "LogAggregation",
    "LogPattern",
    "LogAnomaly",
    "LogQueryBuilder",
    "LogRetentionPolicy",
    
    # Metric Models
    "MetricType",
    "ResultType",
    "TargetHealth",
    "RuleType",
    "RuleHealth",
    "MetricMetadata",
    "Label",
    "Sample",
    "InstantVector",
    "RangeVector",
    "ScalarResult",
    "StringResult",
    "MetricQuery",
    "MetricRangeQuery",
    "MetricResponse",
    "Target",
    "TargetsResponse",
    "Rule",
    "RuleGroup",
    "RulesResponse",
    "PrometheusAlert",
    "AlertsResponse",
    "TSDBStatus",
    "SeriesMetadata",
    "QueryExemplar",
    "BuildInfo",
    "RuntimeInfo",
    "QueryBuilder",
    "MetricAggregation",
    "CommonQueries",
    "SnapshotRequest",
    "SnapshotResponse",
    "DeleteSeriesRequest",
    "MetricStatistics",
    
    # Trace Models
    "SpanKind",
    "SpanStatus",
    "ValueType",
    "ReferenceType",
    "ProcessTag",
    "Process",
    "SpanLog",
    "SpanReference",
    "SpanTag",
    "Span",
    "Trace",
    "TraceSearchParams",
    "TraceSearchResponse",
    "Service",
    "Operation",
    "ServiceDependency",
    "ServiceDependencies",
    "TraceTimeline",
    "TraceMetrics",
    "ServiceMetrics",
    "TraceComparison",
    "LatencyPercentiles",
    "FlamegraphNode",
    "Flamegraph",
    "ServiceNode",
    "ServiceEdge",
    "ServiceMap",
    "SpanKindStats",
    "TraceStatistics",
    "SamplingConfig",
    "StorageInfo",
    "JaegerConfig",
    "TraceDebugInfo",
    "TraceQueryBuilder",
    "SpanContext"
]

# Module version
__version__ = "1.0.0"

# Initialize logging for models module
import logging
logger = logging.getLogger(__name__)
logger.info(f"Models module initialized (version {__version__})")