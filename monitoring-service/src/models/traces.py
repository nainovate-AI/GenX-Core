# monitoring-service/src/models/traces.py
"""Trace models for monitoring service"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator

class SpanKind(str, Enum):
    """Span kind types"""
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"

class SpanStatus(str, Enum):
    """Span status codes"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

class ValueType(str, Enum):
    """Span attribute value types"""
    STRING = "string"
    BOOL = "bool"
    INT64 = "int64"
    FLOAT64 = "float64"
    BINARY = "binary"

class ReferenceType(str, Enum):
    """Span reference types"""
    CHILD_OF = "child_of"
    FOLLOWS_FROM = "follows_from"

class ProcessTag(BaseModel):
    """Process tag"""
    key: str = Field(..., description="Tag key")
    type: ValueType = Field(..., description="Tag value type")
    value: Any = Field(..., description="Tag value")

class Process(BaseModel):
    """Process information"""
    serviceName: str = Field(..., description="Service name")
    tags: List[ProcessTag] = Field(default_factory=list, description="Process tags")
    
    def get_tag(self, key: str) -> Optional[Any]:
        """Get tag value by key"""
        for tag in self.tags:
            if tag.key == key:
                return tag.value
        return None

class SpanLog(BaseModel):
    """Span log entry"""
    timestamp: int = Field(..., description="Log timestamp in microseconds")
    fields: List[Dict[str, Any]] = Field(..., description="Log fields")
    
    @property
    def timestamp_datetime(self) -> datetime:
        """Convert timestamp to datetime"""
        return datetime.fromtimestamp(self.timestamp / 1_000_000)

class SpanReference(BaseModel):
    """Reference to another span"""
    refType: ReferenceType = Field(..., description="Reference type")
    traceID: str = Field(..., description="Trace ID")
    spanID: str = Field(..., description="Span ID")

class SpanTag(BaseModel):
    """Span tag"""
    key: str = Field(..., description="Tag key")
    type: ValueType = Field(..., description="Tag value type")
    value: Any = Field(..., description="Tag value")

class Span(BaseModel):
    """Individual span in a trace"""
    traceID: str = Field(..., description="Trace ID")
    spanID: str = Field(..., description="Span ID")
    operationName: str = Field(..., description="Operation name")
    references: List[SpanReference] = Field(default_factory=list, description="Span references")
    startTime: int = Field(..., description="Start time in microseconds")
    duration: int = Field(..., description="Duration in microseconds")
    tags: List[SpanTag] = Field(default_factory=list, description="Span tags")
    logs: List[SpanLog] = Field(default_factory=list, description="Span logs")
    processID: str = Field(..., description="Process ID")
    warnings: Optional[List[str]] = Field(None, description="Span warnings")
    
    @property
    def start_time_datetime(self) -> datetime:
        """Convert start time to datetime"""
        return datetime.fromtimestamp(self.startTime / 1_000_000)
    
    @property
    def end_time_datetime(self) -> datetime:
        """Calculate end time as datetime"""
        return datetime.fromtimestamp((self.startTime + self.duration) / 1_000_000)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration / 1000
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds"""
        return self.duration / 1_000_000
    
    def get_tag(self, key: str) -> Optional[Any]:
        """Get tag value by key"""
        for tag in self.tags:
            if tag.key == key:
                return tag.value
        return None
    
    def has_error(self) -> bool:
        """Check if span has error"""
        error_tag = self.get_tag("error")
        return error_tag is True or error_tag == "true"
    
    def get_status_code(self) -> Optional[int]:
        """Get HTTP status code if present"""
        status = self.get_tag("http.status_code")
        if status:
            return int(status)
        return None

class Trace(BaseModel):
    """Complete trace with all spans"""
    traceID: str = Field(..., description="Trace ID")
    spans: List[Span] = Field(..., description="All spans in trace")
    processes: Dict[str, Process] = Field(..., description="Process information by ID")
    warnings: Optional[List[str]] = Field(None, description="Trace warnings")
    
    @property
    def duration(self) -> int:
        """Total trace duration in microseconds"""
        if not self.spans:
            return 0
        
        min_start = min(span.startTime for span in self.spans)
        max_end = max(span.startTime + span.duration for span in self.spans)
        return max_end - min_start
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration / 1000
    
    @property
    def start_time(self) -> int:
        """Trace start time in microseconds"""
        if not self.spans:
            return 0
        return min(span.startTime for span in self.spans)
    
    @property
    def start_time_datetime(self) -> datetime:
        """Trace start time as datetime"""
        return datetime.fromtimestamp(self.start_time / 1_000_000)
    
    @property
    def service_names(self) -> List[str]:
        """Get all service names in trace"""
        return list(set(process.serviceName for process in self.processes.values()))
    
    @property
    def span_count(self) -> int:
        """Total number of spans"""
        return len(self.spans)
    
    @property
    def error_count(self) -> int:
        """Number of spans with errors"""
        return sum(1 for span in self.spans if span.has_error())
    
    def get_root_span(self) -> Optional[Span]:
        """Get root span of trace"""
        # Root span has no references or only follows_from references
        for span in self.spans:
            if not span.references or all(ref.refType == ReferenceType.FOLLOWS_FROM for ref in span.references):
                return span
        # Fallback: earliest span
        return min(self.spans, key=lambda s: s.startTime) if self.spans else None
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID"""
        for span in self.spans:
            if span.spanID == span_id:
                return span
        return None
    
    def get_children(self, span_id: str) -> List[Span]:
        """Get child spans of a span"""
        children = []
        for span in self.spans:
            for ref in span.references:
                if ref.spanID == span_id and ref.refType == ReferenceType.CHILD_OF:
                    children.append(span)
                    break
        return children

class TraceSearchParams(BaseModel):
    """Parameters for searching traces"""
    service: str = Field(..., description="Service name")
    operation: Optional[str] = Field(None, description="Operation name")
    tags: Optional[Dict[str, str]] = Field(None, description="Tag filters")
    start_time: Optional[datetime] = Field(None, description="Start time lower bound")
    end_time: Optional[datetime] = Field(None, description="End time upper bound")
    min_duration: Optional[str] = Field(None, description="Minimum duration")
    max_duration: Optional[str] = Field(None, description="Maximum duration")
    limit: int = Field(20, ge=1, le=1000, description="Maximum traces to return")
    
    @validator("min_duration", "max_duration")
    def validate_duration(cls, v):
        """Validate duration format"""
        if v:
            import re
            if not re.match(r'^\d+(\.\d+)?[num]?s$', v):
                raise ValueError("Duration must be in format like 100ms, 1.5s, 1000us")
        return v
    
    def to_query_params(self) -> Dict[str, Any]:
        """Convert to Jaeger query parameters"""
        params = {
            "service": self.service,
            "limit": self.limit
        }
        
        if self.operation:
            params["operation"] = self.operation
        
        if self.tags:
            # Convert tags to Jaeger format
            params["tags"] = "&".join(f"{k}={v}" for k, v in self.tags.items())
        
        if self.start_time:
            params["start"] = int(self.start_time.timestamp() * 1_000_000)
        
        if self.end_time:
            params["end"] = int(self.end_time.timestamp() * 1_000_000)
        
        if self.min_duration:
            params["minDuration"] = self.min_duration
        
        if self.max_duration:
            params["maxDuration"] = self.max_duration
        
        return params

class TraceSearchResponse(BaseModel):
    """Response from trace search"""
    data: List[Trace] = Field(..., description="Found traces")
    total: int = Field(..., description="Total number of matching traces")
    limit: int = Field(..., description="Limit used in search")
    offset: int = Field(0, description="Offset used in search")
    errors: Optional[List[str]] = Field(None, description="Any errors during search")

class Service(BaseModel):
    """Service information"""
    name: str = Field(..., description="Service name")
    operation_count: Optional[int] = Field(None, description="Number of operations")

class Operation(BaseModel):
    """Operation information"""
    name: str = Field(..., description="Operation name")
    service: str = Field(..., description="Service name")

class ServiceDependency(BaseModel):
    """Service dependency relationship"""
    parent: str = Field(..., description="Parent service name")
    child: str = Field(..., description="Child service name")
    callCount: int = Field(..., description="Number of calls")
    
class ServiceDependencies(BaseModel):
    """Service dependencies graph"""
    dependencies: List[ServiceDependency] = Field(..., description="All dependencies")
    errors: Optional[List[str]] = Field(None, description="Any errors")
    
    def get_services(self) -> List[str]:
        """Get all unique services"""
        services = set()
        for dep in self.dependencies:
            services.add(dep.parent)
            services.add(dep.child)
        return sorted(list(services))
    
    def get_dependencies_for_service(self, service: str) -> Dict[str, List[str]]:
        """Get upstream and downstream dependencies for a service"""
        upstream = []
        downstream = []
        
        for dep in self.dependencies:
            if dep.child == service:
                upstream.append(dep.parent)
            elif dep.parent == service:
                downstream.append(dep.child)
        
        return {
            "upstream": sorted(list(set(upstream))),
            "downstream": sorted(list(set(downstream)))
        }

class TraceTimeline(BaseModel):
    """Timeline visualization data for a trace"""
    trace_id: str = Field(..., description="Trace ID")
    duration_ms: float = Field(..., description="Total duration in milliseconds")
    service_timelines: Dict[str, List[Dict[str, Any]]] = Field(..., description="Timeline per service")
    critical_path: List[str] = Field(..., description="Span IDs in critical path")
    
class TraceMetrics(BaseModel):
    """Aggregated metrics for a trace"""
    trace_id: str = Field(..., description="Trace ID")
    total_spans: int = Field(..., description="Total number of spans")
    duration_ms: float = Field(..., description="Total duration in milliseconds")
    services_involved: int = Field(..., description="Number of services")
    error_count: int = Field(..., description="Number of errors")
    by_service: Dict[str, Dict[str, Any]] = Field(..., description="Metrics grouped by service")
    by_operation: Dict[str, Dict[str, Any]] = Field(..., description="Metrics grouped by operation")

class ServiceMetrics(BaseModel):
    """Aggregated metrics for a service"""
    service: str = Field(..., description="Service name")
    operation_count: int = Field(..., description="Number of operations")
    trace_count: int = Field(..., description="Number of traces")
    error_rate: float = Field(..., description="Error rate percentage")
    avg_duration_ms: float = Field(..., description="Average duration in milliseconds")
    p50_duration_ms: float = Field(..., description="P50 duration in milliseconds")
    p95_duration_ms: float = Field(..., description="P95 duration in milliseconds")
    p99_duration_ms: float = Field(..., description="P99 duration in milliseconds")

class TraceComparison(BaseModel):
    """Comparison between multiple traces"""
    trace_ids: List[str] = Field(..., description="Trace IDs being compared")
    common_operations: List[str] = Field(..., description="Operations present in all traces")
    duration_comparison: Dict[str, float] = Field(..., description="Duration by trace ID")
    span_count_comparison: Dict[str, int] = Field(..., description="Span count by trace ID")
    service_comparison: Dict[str, List[str]] = Field(..., description="Services by trace ID")
    operation_timings: Dict[str, Dict[str, float]] = Field(..., description="Operation timings comparison")

class LatencyPercentiles(BaseModel):
    """Latency percentile data"""
    service: str = Field(..., description="Service name")
    operation: Optional[str] = Field(None, description="Operation name")
    percentiles: Dict[str, float] = Field(..., description="Percentile values in milliseconds")
    sample_count: int = Field(..., description="Number of samples")
    time_range: Dict[str, datetime] = Field(..., description="Time range of data")

class FlamegraphNode(BaseModel):
    """Flamegraph node data"""
    name: str = Field(..., description="Node name (operation)")
    value: int = Field(..., description="Duration in microseconds")
    children: List["FlamegraphNode"] = Field(default_factory=list, description="Child nodes")
    
FlamegraphNode.model_rebuild()

class Flamegraph(BaseModel):
    """Flamegraph data for trace visualization"""
    trace_id: str = Field(..., description="Trace ID")
    root: FlamegraphNode = Field(..., description="Root node of flamegraph")
    total_time: int = Field(..., description="Total time in microseconds")

class ServiceNode(BaseModel):
    """Node in service dependency graph"""
    id: str = Field(..., description="Service name")
    label: str = Field(..., description="Display label")
    metrics: Optional[ServiceMetrics] = Field(None, description="Service metrics")

class ServiceEdge(BaseModel):
    """Edge in service dependency graph"""
    source: str = Field(..., description="Source service")
    target: str = Field(..., description="Target service")
    call_count: int = Field(..., description="Number of calls")
    error_count: Optional[int] = Field(0, description="Number of errors")
    avg_duration_ms: Optional[float] = Field(None, description="Average call duration")

class ServiceMap(BaseModel):
    """Service dependency map"""
    nodes: List[ServiceNode] = Field(..., description="Service nodes")
    edges: List[ServiceEdge] = Field(..., description="Service relationships")
    depth: int = Field(..., description="Depth of dependencies included")
    timestamp: datetime = Field(..., description="Map generation time")

class SpanKindStats(BaseModel):
    """Statistics grouped by span kind"""
    client: int = Field(0, description="Client spans")
    server: int = Field(0, description="Server spans")
    producer: int = Field(0, description="Producer spans")
    consumer: int = Field(0, description="Consumer spans")
    internal: int = Field(0, description="Internal spans")

class TraceStatistics(BaseModel):
    """Aggregate statistics for traces"""
    total_traces: int = Field(..., description="Total number of traces")
    total_spans: int = Field(..., description="Total number of spans")
    services: List[str] = Field(..., description="All services")
    operations: List[str] = Field(..., description="All operations")
    span_kind_stats: SpanKindStats = Field(..., description="Spans by kind")
    error_traces: int = Field(..., description="Traces with errors")
    avg_spans_per_trace: float = Field(..., description="Average spans per trace")
    avg_duration_ms: float = Field(..., description="Average trace duration")

class SamplingConfig(BaseModel):
    """Trace sampling configuration"""
    type: str = Field(..., description="Sampling type: probabilistic, ratelimiting, adaptive")
    param: float = Field(..., description="Sampling parameter")
    max_traces_per_second: Optional[float] = Field(None, description="Rate limit")

class StorageInfo(BaseModel):
    """Jaeger storage information"""
    span_storage_type: str = Field(..., description="Storage type: memory, cassandra, elasticsearch, etc.")
    dependencies_storage_type: Optional[str] = Field(None, description="Dependencies storage type")

class JaegerConfig(BaseModel):
    """Jaeger configuration information"""
    version: str = Field(..., description="Jaeger version")
    storage: StorageInfo = Field(..., description="Storage configuration")
    sampling: Optional[SamplingConfig] = Field(None, description="Sampling configuration")
    flags: Dict[str, Any] = Field(..., description="Feature flags")

class TraceDebugInfo(BaseModel):
    """Debug information for trace analysis"""
    trace_id: str = Field(..., description="Trace ID")
    warnings: List[str] = Field(..., description="Trace warnings")
    missing_spans: List[str] = Field(..., description="Potentially missing span IDs")
    clock_skew_detected: bool = Field(..., description="Whether clock skew was detected")
    orphaned_spans: List[str] = Field(..., description="Spans without parents")
    circular_references: List[str] = Field(..., description="Spans with circular references")

class TraceQueryBuilder(BaseModel):
    """Helper for building trace queries"""
    service: str = Field(..., description="Service name")
    operation: Optional[str] = Field(None, description="Operation name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tag filters")
    min_duration_ms: Optional[float] = Field(None, description="Minimum duration in ms")
    max_duration_ms: Optional[float] = Field(None, description="Maximum duration in ms")
    
    def build_tags_query(self) -> str:
        """Build tags query string for Jaeger"""
        tag_parts = []
        for key, value in self.tags.items():
            tag_parts.append(f"{key}={value}")
        return "&".join(tag_parts)
    
    def to_search_params(self) -> TraceSearchParams:
        """Convert to TraceSearchParams"""
        params = TraceSearchParams(service=self.service)
        
        if self.operation:
            params.operation = self.operation
        
        if self.tags:
            params.tags = self.tags
        
        if self.min_duration_ms:
            params.min_duration = f"{int(self.min_duration_ms)}ms"
        
        if self.max_duration_ms:
            params.max_duration = f"{int(self.max_duration_ms)}ms"
        
        return params

class SpanContext(BaseModel):
    """Span context for distributed tracing"""
    trace_id: str = Field(..., description="Trace ID")
    span_id: str = Field(..., description="Span ID")
    trace_flags: int = Field(0, description="Trace flags")
    trace_state: Optional[str] = Field(None, description="Trace state")
    is_remote: bool = Field(False, description="Whether span is remote")