# monitoring-service/src/models/metrics.py
"""Metric models for monitoring service"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator

class MetricType(str, Enum):
    """Prometheus metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    UNTYPED = "untyped"

class ResultType(str, Enum):
    """Prometheus query result types"""
    MATRIX = "matrix"
    VECTOR = "vector"
    SCALAR = "scalar"
    STRING = "string"

class TargetHealth(str, Enum):
    """Target health status"""
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"

class RuleType(str, Enum):
    """Prometheus rule types"""
    RECORDING = "recording"
    ALERTING = "alerting"

class RuleHealth(str, Enum):
    """Rule health status"""
    OK = "ok"
    ERROR = "err"
    UNKNOWN = "unknown"

class MetricMetadata(BaseModel):
    """Metric metadata information"""
    metric: str = Field(..., description="Metric name")
    type: MetricType = Field(..., description="Metric type")
    help: str = Field(..., description="Metric help text")
    unit: Optional[str] = Field(None, description="Metric unit")

class Label(BaseModel):
    """Label key-value pair"""
    name: str = Field(..., description="Label name")
    value: str = Field(..., description="Label value")

class Sample(BaseModel):
    """Single metric sample"""
    timestamp: float = Field(..., description="Unix timestamp")
    value: Union[float, str] = Field(..., description="Sample value")
    
    @validator("value")
    def validate_value(cls, v):
        """Validate sample value"""
        if isinstance(v, str):
            # Check for special float values
            if v not in ["NaN", "+Inf", "-Inf"]:
                try:
                    return float(v)
                except ValueError:
                    pass
        return v

class InstantVector(BaseModel):
    """Instant vector result"""
    metric: Dict[str, str] = Field(..., description="Metric labels")
    value: Sample = Field(..., description="Single sample value")
    
    @validator("value", pre=True)
    def parse_value(cls, v):
        """Parse value array to Sample"""
        if isinstance(v, list) and len(v) == 2:
            return Sample(timestamp=v[0], value=v[1])
        return v

class RangeVector(BaseModel):
    """Range vector result"""
    metric: Dict[str, str] = Field(..., description="Metric labels")
    values: List[Sample] = Field(..., description="Time series samples")
    
    @validator("values", pre=True)
    def parse_values(cls, v):
        """Parse values array to Sample list"""
        if isinstance(v, list) and all(isinstance(item, list) for item in v):
            return [Sample(timestamp=item[0], value=item[1]) for item in v]
        return v

class ScalarResult(BaseModel):
    """Scalar query result"""
    timestamp: float = Field(..., description="Result timestamp")
    value: Union[float, str] = Field(..., description="Scalar value")

class StringResult(BaseModel):
    """String query result"""
    timestamp: float = Field(..., description="Result timestamp")
    value: str = Field(..., description="String value")

class MetricQuery(BaseModel):
    """Metric query parameters"""
    query: str = Field(..., description="PromQL query")
    time: Optional[datetime] = Field(None, description="Evaluation timestamp")
    timeout: Optional[str] = Field("30s", description="Query timeout")
    
    @validator("query")
    def validate_query(cls, v):
        """Basic PromQL validation"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class MetricRangeQuery(BaseModel):
    """Metric range query parameters"""
    query: str = Field(..., description="PromQL query")
    start: datetime = Field(..., description="Start timestamp")
    end: datetime = Field(..., description="End timestamp")
    step: str = Field("15s", description="Query resolution step")
    timeout: Optional[str] = Field("30s", description="Query timeout")
    
    @validator("step")
    def validate_step(cls, v):
        """Validate step format"""
        import re
        if not re.match(r'^\d+[smhd]$', v):
            raise ValueError("Step must be in format like 15s, 5m, 1h")
        return v
    
    @root_validator
    def validate_time_range(cls, values):
        """Validate time range"""
        start = values.get("start")
        end = values.get("end")
        if start and end and start >= end:
            raise ValueError("Start time must be before end time")
        return values

class MetricResponse(BaseModel):
    """Generic metric query response"""
    status: str = Field("success", description="Query status")
    data: Dict[str, Any] = Field(..., description="Query result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    errorType: Optional[str] = Field(None, description="Error type if failed")
    warnings: Optional[List[str]] = Field(None, description="Query warnings")
    
    @property
    def result_type(self) -> Optional[ResultType]:
        """Get result type from data"""
        if self.data and "resultType" in self.data:
            return ResultType(self.data["resultType"])
        return None
    
    @property
    def result(self) -> Union[List[InstantVector], List[RangeVector], ScalarResult, StringResult, None]:
        """Get typed result based on result type"""
        if not self.data or "result" not in self.data:
            return None
        
        result_type = self.result_type
        result_data = self.data["result"]
        
        if result_type == ResultType.VECTOR:
            return [InstantVector(**item) for item in result_data]
        elif result_type == ResultType.MATRIX:
            return [RangeVector(**item) for item in result_data]
        elif result_type == ResultType.SCALAR:
            if isinstance(result_data, list) and len(result_data) == 2:
                return ScalarResult(timestamp=result_data[0], value=result_data[1])
        elif result_type == ResultType.STRING:
            if isinstance(result_data, list) and len(result_data) == 2:
                return StringResult(timestamp=result_data[0], value=result_data[1])
        
        return None

class Target(BaseModel):
    """Prometheus scrape target"""
    discoveredLabels: Dict[str, str] = Field(..., description="Discovered labels")
    labels: Dict[str, str] = Field(..., description="Target labels after relabeling")
    scrapePool: str = Field(..., description="Scrape pool name")
    scrapeUrl: str = Field(..., description="URL being scraped")
    globalUrl: str = Field(..., description="Global URL")
    lastError: str = Field("", description="Last scrape error")
    lastScrape: datetime = Field(..., description="Last scrape timestamp")
    lastScrapeDuration: float = Field(..., description="Last scrape duration in seconds")
    health: TargetHealth = Field(..., description="Target health status")
    scrapeInterval: str = Field(..., description="Scrape interval")
    scrapeTimeout: str = Field(..., description="Scrape timeout")

class TargetsResponse(BaseModel):
    """Targets query response"""
    status: str = Field("success", description="Query status")
    data: Dict[str, List[Target]] = Field(..., description="Targets grouped by state")
    
    @property
    def active_targets(self) -> List[Target]:
        """Get active targets"""
        return self.data.get("activeTargets", [])
    
    @property
    def dropped_targets(self) -> List[Dict[str, Any]]:
        """Get dropped targets"""
        return self.data.get("droppedTargets", [])

class Rule(BaseModel):
    """Prometheus rule (recording or alerting)"""
    name: str = Field(..., description="Rule name")
    query: str = Field(..., description="Rule query expression")
    type: RuleType = Field(..., description="Rule type")
    health: RuleHealth = Field(..., description="Rule health")
    evaluationTime: float = Field(..., description="Last evaluation duration")
    lastEvaluation: datetime = Field(..., description="Last evaluation timestamp")
    labels: Optional[Dict[str, str]] = Field(None, description="Rule labels")
    annotations: Optional[Dict[str, str]] = Field(None, description="Rule annotations")
    duration: Optional[float] = Field(None, description="Duration for alerting rules")
    keepFiringFor: Optional[float] = Field(None, description="Keep firing duration")
    state: Optional[str] = Field(None, description="Alert state")
    alerts: Optional[List[Dict[str, Any]]] = Field(None, description="Active alerts for this rule")
    lastError: Optional[str] = Field(None, description="Last evaluation error")

class RuleGroup(BaseModel):
    """Group of Prometheus rules"""
    name: str = Field(..., description="Group name")
    file: str = Field(..., description="Rule file path")
    interval: float = Field(..., description="Evaluation interval in seconds")
    evaluationTime: float = Field(..., description="Last evaluation duration")
    lastEvaluation: datetime = Field(..., description="Last evaluation timestamp")
    rules: List[Rule] = Field(..., description="Rules in this group")
    
    @property
    def recording_rules(self) -> List[Rule]:
        """Get recording rules"""
        return [r for r in self.rules if r.type == RuleType.RECORDING]
    
    @property
    def alerting_rules(self) -> List[Rule]:
        """Get alerting rules"""
        return [r for r in self.rules if r.type == RuleType.ALERTING]

class RulesResponse(BaseModel):
    """Rules query response"""
    status: str = Field("success", description="Query status")
    data: Dict[str, List[RuleGroup]] = Field(..., description="Rule groups")
    
    @property
    def groups(self) -> List[RuleGroup]:
        """Get all rule groups"""
        return self.data.get("groups", [])

class Alert(BaseModel):
    """Active alert from Prometheus"""
    labels: Dict[str, str] = Field(..., description="Alert labels")
    annotations: Dict[str, str] = Field(..., description="Alert annotations")
    state: str = Field(..., description="Alert state: pending or firing")
    activeAt: datetime = Field(..., description="When alert became active")
    value: str = Field(..., description="Alert value")
    keepFiringSince: Optional[datetime] = Field(None, description="Keep firing since")

class AlertsResponse(BaseModel):
    """Alerts query response"""
    status: str = Field("success", description="Query status")
    data: Dict[str, List[Alert]] = Field(..., description="Alerts data")
    
    @property
    def alerts(self) -> List[Alert]:
        """Get all alerts"""
        return self.data.get("alerts", [])

class TSDBStatus(BaseModel):
    """TSDB (Time Series Database) status"""
    headStats: Dict[str, Any] = Field(..., description="Head block statistics")
    seriesCountByMetricName: List[Dict[str, Any]] = Field(..., description="Series count by metric")
    labelValueCountByLabelName: List[Dict[str, Any]] = Field(..., description="Label value count")
    memoryInBytesByLabelName: List[Dict[str, Any]] = Field(..., description="Memory usage by label")
    seriesCountByLabelValuePair: List[Dict[str, Any]] = Field(..., description="Series count by label pair")

class SeriesMetadata(BaseModel):
    """Time series metadata"""
    labels: Dict[str, str] = Field(..., description="Series labels")
    metric: Optional[str] = Field(None, description="Metric name")
    
    @validator("metric", pre=True, always=True)
    def extract_metric(cls, v, values):
        """Extract metric name from labels"""
        if not v and "labels" in values:
            return values["labels"].get("__name__")
        return v

class QueryExemplar(BaseModel):
    """Query exemplar (trace ID)"""
    labels: Dict[str, str] = Field(..., description="Exemplar labels")
    value: float = Field(..., description="Exemplar value")
    timestamp: float = Field(..., description="Exemplar timestamp")
    traceID: Optional[str] = Field(None, description="Associated trace ID")

class BuildInfo(BaseModel):
    """Prometheus build information"""
    version: str = Field(..., description="Prometheus version")
    revision: str = Field(..., description="Git revision")
    branch: str = Field(..., description="Git branch")
    buildUser: str = Field(..., description="Build user")
    buildDate: str = Field(..., description="Build date")
    goVersion: str = Field(..., description="Go version")

class RuntimeInfo(BaseModel):
    """Prometheus runtime information"""
    startTime: datetime = Field(..., description="Process start time")
    CWD: str = Field(..., description="Current working directory")
    reloadConfigSuccess: bool = Field(..., description="Last config reload success")
    lastConfigTime: datetime = Field(..., description="Last config reload time")
    corruptionCount: int = Field(..., description="TSDB corruption count")
    goroutineCount: int = Field(..., description="Number of goroutines")
    GOMAXPROCS: int = Field(..., description="GOMAXPROCS value")
    GOMEMLIMIT: Optional[int] = Field(None, description="GOMEMLIMIT value")
    GOGC: str = Field(..., description="GOGC value")
    GODEBUG: str = Field(..., description="GODEBUG value")
    storageRetention: str = Field(..., description="Storage retention")

class QueryBuilder(BaseModel):
    """Helper for building PromQL queries"""
    metric: str = Field(..., description="Metric name")
    labels: Optional[Dict[str, str]] = Field(None, description="Label matchers")
    function: Optional[str] = Field(None, description="Function to apply")
    range_duration: Optional[str] = Field(None, description="Range duration for range vectors")
    aggregation: Optional[str] = Field(None, description="Aggregation operator")
    by_labels: Optional[List[str]] = Field(None, description="Labels for by clause")
    without_labels: Optional[List[str]] = Field(None, description="Labels for without clause")
    
    def build(self) -> str:
        """Build PromQL query string"""
        # Base metric with labels
        label_matchers = []
        if self.labels:
            for key, value in self.labels.items():
                if "*" in value:
                    label_matchers.append(f'{key}=~"{value}"')
                else:
                    label_matchers.append(f'{key}="{value}"')
        
        if label_matchers:
            query = f"{self.metric}{{{','.join(label_matchers)}}}"
        else:
            query = self.metric
        
        # Add range duration
        if self.range_duration:
            query = f"{query}[{self.range_duration}]"
        
        # Apply function
        if self.function:
            query = f"{self.function}({query})"
        
        # Apply aggregation
        if self.aggregation:
            if self.by_labels:
                query = f"{self.aggregation} by ({','.join(self.by_labels)}) ({query})"
            elif self.without_labels:
                query = f"{self.aggregation} without ({','.join(self.without_labels)}) ({query})"
            else:
                query = f"{self.aggregation}({query})"
        
        return query

class MetricAggregation(BaseModel):
    """Common metric aggregation parameters"""
    operation: str = Field(..., description="Aggregation operation: sum, avg, min, max, etc.")
    by: Optional[List[str]] = Field(None, description="Group by labels")
    without: Optional[List[str]] = Field(None, description="Exclude labels")
    keep_common: bool = Field(False, description="Keep common labels")
    
    def apply_to_query(self, query: str) -> str:
        """Apply aggregation to a query"""
        if self.by:
            return f"{self.operation} by ({','.join(self.by)}) ({query})"
        elif self.without:
            return f"{self.operation} without ({','.join(self.without)}) ({query})"
        else:
            return f"{self.operation}({query})"

class CommonQueries(BaseModel):
    """Common Prometheus queries"""
    
    @staticmethod
    def cpu_usage(instance: Optional[str] = None) -> str:
        """CPU usage percentage query"""
        base = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))'
        if instance:
            base = base.replace("{", f'{{instance="{instance}",')
        return f"{base} * 100)"
    
    @staticmethod
    def memory_usage(instance: Optional[str] = None) -> str:
        """Memory usage percentage query"""
        if instance:
            return f'(1 - (node_memory_MemAvailable_bytes{{instance="{instance}"}} / node_memory_MemTotal_bytes{{instance="{instance}"}})) * 100'
        return '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
    
    @staticmethod
    def disk_usage(mountpoint: str = "/", instance: Optional[str] = None) -> str:
        """Disk usage percentage query"""
        labels = f'mountpoint="{mountpoint}"'
        if instance:
            labels += f',instance="{instance}"'
        return f'100 - ((node_filesystem_avail_bytes{{{labels}}} * 100) / node_filesystem_size_bytes{{{labels}}})'
    
    @staticmethod
    def request_rate(job: str, status_code: Optional[str] = None) -> str:
        """HTTP request rate query"""
        labels = f'job="{job}"'
        if status_code:
            labels += f',status=~"{status_code}"'
        return f'sum(rate(http_requests_total{{{labels}}}[5m])) by (job, status)'
    
    @staticmethod
    def error_rate(job: str) -> str:
        """HTTP error rate query"""
        return f'sum(rate(http_requests_total{{job="{job}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{job="{job}"}}[5m]))'
    
    @staticmethod
    def p95_latency(job: str) -> str:
        """95th percentile latency query"""
        return f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{job="{job}"}}[5m])) by (le))'

class SnapshotRequest(BaseModel):
    """Request to create TSDB snapshot"""
    skip_head: bool = Field(False, description="Skip head block in snapshot")

class SnapshotResponse(BaseModel):
    """Response from snapshot creation"""
    status: str = Field("success", description="Operation status")
    data: Dict[str, str] = Field(..., description="Snapshot data with name")

class DeleteSeriesRequest(BaseModel):
    """Request to delete time series"""
    matchers: List[str] = Field(..., description="Series selectors")
    start: datetime = Field(..., description="Start time for deletion")
    end: datetime = Field(..., description="End time for deletion")
    
    @validator("matchers")
    def validate_matchers(cls, v):
        """Validate at least one matcher provided"""
        if not v:
            raise ValueError("At least one matcher is required")
        return v

class MetricStatistics(BaseModel):
    """Statistics for a specific metric"""
    metric_name: str = Field(..., description="Metric name")
    series_count: int = Field(..., description="Number of series")
    samples_count: int = Field(..., description="Number of samples")
    oldest_sample: Optional[datetime] = Field(None, description="Oldest sample timestamp")
    newest_sample: Optional[datetime] = Field(None, description="Newest sample timestamp")
    label_cardinality: Dict[str, int] = Field(..., description="Cardinality per label")
    memory_bytes: int = Field(..., description="Memory usage in bytes")