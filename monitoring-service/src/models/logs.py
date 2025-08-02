# monitoring-service/src/models/logs.py
"""Log models for monitoring service"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class LogLevel(str, Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRACE = "trace"
    FATAL = "fatal"

class LogDirection(str, Enum):
    """Log query direction"""
    FORWARD = "forward"
    BACKWARD = "backward"

class LogEntry(BaseModel):
    """Individual log entry"""
    timestamp: datetime = Field(..., description="Log timestamp")
    line: str = Field(..., description="Log line content")
    labels: Dict[str, str] = Field(default_factory=dict, description="Log labels")
    level: Optional[LogLevel] = Field(None, description="Log level if parsed")
    
    @validator("timestamp", pre=True)
    def parse_timestamp(cls, v):
        """Parse timestamp from various formats"""
        if isinstance(v, str):
            # Try parsing nanosecond timestamp
            if v.isdigit() and len(v) > 13:
                # Nanosecond timestamp
                return datetime.fromtimestamp(int(v) / 1e9)
            # Try ISO format
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class LogStream(BaseModel):
    """Log stream with labels and entries"""
    stream: Dict[str, str] = Field(..., description="Stream labels")
    values: List[List[str]] = Field(..., description="Log entries [timestamp, line]")
    
    def to_entries(self) -> List[LogEntry]:
        """Convert to LogEntry list"""
        entries = []
        for timestamp, line in self.values:
            entries.append(LogEntry(
                timestamp=timestamp,
                line=line,
                labels=self.stream
            ))
        return entries

class LogQuery(BaseModel):
    """Log query parameters"""
    query: str = Field(..., description="LogQL query string")
    start: Optional[datetime] = Field(None, description="Start time")
    end: Optional[datetime] = Field(None, description="End time")
    limit: int = Field(100, ge=1, le=5000, description="Maximum entries to return")
    direction: LogDirection = Field(LogDirection.BACKWARD, description="Query direction")
    step: Optional[str] = Field(None, description="Query resolution step for metric queries")
    interval: Optional[str] = Field(None, description="Return entries at specific interval")
    
    @validator("query")
    def validate_query(cls, v):
        """Basic LogQL validation"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if not v.startswith("{"):
            raise ValueError("LogQL query must start with a stream selector {}")
        return v

class LogQueryResponse(BaseModel):
    """Response from log query"""
    status: str = Field(..., description="Query status")
    data: Dict[str, Any] = Field(..., description="Query result data")
    
    @property
    def result_type(self) -> str:
        """Get result type from data"""
        return self.data.get("resultType", "unknown")
    
    @property
    def streams(self) -> List[LogStream]:
        """Get log streams from result"""
        if self.result_type == "streams":
            return [LogStream(**stream) for stream in self.data.get("result", [])]
        return []
    
    @property
    def entries(self) -> List[LogEntry]:
        """Get all log entries flattened"""
        entries = []
        for stream in self.streams:
            entries.extend(stream.to_entries())
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)

class LogStreamResponse(BaseModel):
    """Response for log streaming"""
    streams: List[LogStream] = Field(..., description="Log streams")
    dropped_entries: Optional[List[Dict[str, Any]]] = Field(None, description="Dropped entries due to limits")
    
    @property
    def total_entries(self) -> int:
        """Total number of entries across all streams"""
        return sum(len(stream.values) for stream in self.streams)

class LogLabelsResponse(BaseModel):
    """Response for label names query"""
    status: str = Field(..., description="Query status")
    data: List[str] = Field(..., description="Label names")

class LogLabelValuesResponse(BaseModel):
    """Response for label values query"""
    status: str = Field(..., description="Query status")
    data: List[str] = Field(..., description="Label values")

class LogSeriesResponse(BaseModel):
    """Response for series query"""
    status: str = Field(..., description="Query status")
    data: List[Dict[str, str]] = Field(..., description="Series data")

class LogMetricQuery(BaseModel):
    """Log metric query (for LogQL metric queries)"""
    query: str = Field(..., description="LogQL metric query")
    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")
    step: str = Field("60s", description="Query resolution step")
    
    @validator("query")
    def validate_metric_query(cls, v):
        """Validate LogQL metric query"""
        if not any(op in v for op in ["rate(", "count_over_time(", "sum(", "avg(", "max(", "min("]):
            raise ValueError("Not a valid LogQL metric query")
        return v

class LogMetricResponse(BaseModel):
    """Response for log metric query"""
    status: str = Field(..., description="Query status")
    data: Dict[str, Any] = Field(..., description="Metric result data")
    
    @property
    def result_type(self) -> str:
        """Get result type"""
        return self.data.get("resultType", "unknown")
    
    @property
    def metrics(self) -> List[Dict[str, Any]]:
        """Get metric results"""
        return self.data.get("result", [])

class LogStats(BaseModel):
    """Log statistics"""
    ingested_lines: int = Field(..., description="Total ingested log lines")
    ingested_bytes: int = Field(..., description="Total ingested bytes")
    compressed_bytes: int = Field(..., description="Compressed storage bytes")
    streams_count: int = Field(..., description="Number of active streams")
    chunks_count: int = Field(..., description="Number of chunks")
    entries_per_second: float = Field(..., description="Current ingestion rate")
    bytes_per_second: float = Field(..., description="Current byte ingestion rate")

class LogStreamStats(BaseModel):
    """Statistics for a specific log stream"""
    stream: Dict[str, str] = Field(..., description="Stream labels")
    entries_count: int = Field(..., description="Number of entries")
    bytes_size: int = Field(..., description="Total bytes")
    first_entry_time: datetime = Field(..., description="First entry timestamp")
    last_entry_time: datetime = Field(..., description="Last entry timestamp")
    ingestion_rate: float = Field(..., description="Entries per second")

class LogTailRequest(BaseModel):
    """Request for tailing logs"""
    query: str = Field(..., description="LogQL query")
    delay_for: int = Field(0, description="Delay in seconds to avoid catching up")
    limit: int = Field(100, description="Maximum entries per response")
    start: Optional[datetime] = Field(None, description="Start from timestamp")

class LogContextRequest(BaseModel):
    """Request for log context (lines before/after)"""
    query: str = Field(..., description="LogQL query to find log")
    timestamp: datetime = Field(..., description="Timestamp of target log")
    before: int = Field(10, ge=0, le=100, description="Lines before")
    after: int = Field(10, ge=0, le=100, description="Lines after")

class LogContextResponse(BaseModel):
    """Response with log context"""
    target_entry: LogEntry = Field(..., description="The target log entry")
    before_entries: List[LogEntry] = Field(..., description="Entries before target")
    after_entries: List[LogEntry] = Field(..., description="Entries after target")

class LogFilter(BaseModel):
    """Log filter criteria"""
    services: Optional[List[str]] = Field(None, description="Filter by service names")
    levels: Optional[List[LogLevel]] = Field(None, description="Filter by log levels")
    search_text: Optional[str] = Field(None, description="Search in log content")
    labels: Optional[Dict[str, str]] = Field(None, description="Filter by labels")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    
    def to_logql(self) -> str:
        """Convert filter to LogQL query"""
        # Build label matchers
        matchers = []
        
        if self.services:
            service_matcher = "|".join(self.services)
            matchers.append(f'service=~"{service_matcher}"')
        
        if self.levels:
            level_matcher = "|".join([level.value for level in self.levels])
            matchers.append(f'level=~"{level_matcher}"')
        
        if self.labels:
            for key, value in self.labels.items():
                matchers.append(f'{key}="{value}"')
        
        # Default matcher if none specified
        if not matchers:
            matchers.append('job=~".+"')
        
        query = "{" + ",".join(matchers) + "}"
        
        # Add line filters
        if self.search_text:
            query += f' |= "{self.search_text}"'
        
        return query

class LogExportRequest(BaseModel):
    """Request for log export"""
    query: str = Field(..., description="LogQL query")
    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")
    format: str = Field("json", description="Export format: json, csv, txt")
    limit: int = Field(10000, description="Maximum entries to export")
    include_labels: bool = Field(True, description="Include labels in export")

class LogIngestionRequest(BaseModel):
    """Request to push logs to Loki"""
    streams: List[Dict[str, Any]] = Field(..., description="Log streams to push")
    
    @validator("streams")
    def validate_streams(cls, v):
        """Validate stream format"""
        for stream in v:
            if "stream" not in stream or "values" not in stream:
                raise ValueError("Each stream must have 'stream' and 'values' fields")
            if not isinstance(stream["stream"], dict):
                raise ValueError("'stream' must be a dictionary of labels")
            if not isinstance(stream["values"], list):
                raise ValueError("'values' must be a list of [timestamp, line] pairs")
        return v

class LogParseRule(BaseModel):
    """Rule for parsing log lines"""
    name: str = Field(..., description="Rule name")
    pattern: str = Field(..., description="Regex pattern or parser expression")
    parser_type: str = Field("regex", description="Parser type: regex, json, logfmt")
    extract_fields: List[str] = Field(..., description="Fields to extract")
    enabled: bool = Field(True, description="Whether rule is enabled")

class LogAggregation(BaseModel):
    """Log aggregation result"""
    group_by: Dict[str, str] = Field(..., description="Grouping labels")
    count: int = Field(..., description="Number of logs in group")
    first_seen: datetime = Field(..., description="First occurrence")
    last_seen: datetime = Field(..., description="Last occurrence")
    sample_messages: List[str] = Field(..., description="Sample log messages")
    
class LogPattern(BaseModel):
    """Detected log pattern"""
    pattern: str = Field(..., description="Log pattern with placeholders")
    count: int = Field(..., description="Number of matches")
    percentage: float = Field(..., description="Percentage of total logs")
    examples: List[str] = Field(..., description="Example log lines")
    variables: List[str] = Field(..., description="Variable parts in pattern")

class LogAnomaly(BaseModel):
    """Detected log anomaly"""
    timestamp: datetime = Field(..., description="Anomaly timestamp")
    score: float = Field(..., description="Anomaly score (0-1)")
    reason: str = Field(..., description="Reason for anomaly")
    log_entry: LogEntry = Field(..., description="Anomalous log entry")
    baseline_stats: Dict[str, Any] = Field(..., description="Baseline statistics")

class LogQueryBuilder(BaseModel):
    """Helper for building LogQL queries"""
    stream_selector: Dict[str, str] = Field(..., description="Label selectors")
    line_filters: List[str] = Field(default_factory=list, description="Line filter expressions")
    label_filters: List[str] = Field(default_factory=list, description="Label filter expressions")
    parser: Optional[str] = Field(None, description="Parser stage: json, logfmt, pattern, regex")
    unwrap: Optional[str] = Field(None, description="Unwrap expression for metrics")
    aggregation: Optional[str] = Field(None, description="Aggregation function")
    
    def build(self) -> str:
        """Build LogQL query string"""
        # Stream selector
        selector_parts = []
        for key, value in self.stream_selector.items():
            if "*" in value or ".*" in value:
                selector_parts.append(f'{key}=~"{value}"')
            else:
                selector_parts.append(f'{key}="{value}"')
        
        query = "{" + ",".join(selector_parts) + "}"
        
        # Line filters
        for filter_expr in self.line_filters:
            query += f" {filter_expr}"
        
        # Parser stage
        if self.parser:
            query += f" | {self.parser}"
        
        # Label filters
        for filter_expr in self.label_filters:
            query += f" | {filter_expr}"
        
        # Unwrap for metrics
        if self.unwrap:
            query += f" | unwrap {self.unwrap}"
        
        # Aggregation
        if self.aggregation:
            query = f"{self.aggregation}({query})"
        
        return query

class LogRetentionPolicy(BaseModel):
    """Log retention policy"""
    stream_selector: Dict[str, str] = Field(..., description="Stream selector for policy")
    retention_period: str = Field(..., description="Retention period (e.g., 7d, 30d)")
    compact_period: Optional[str] = Field(None, description="Compaction period")
    priority: int = Field(0, description="Policy priority (higher wins)")
    enabled: bool = Field(True, description="Whether policy is enabled")