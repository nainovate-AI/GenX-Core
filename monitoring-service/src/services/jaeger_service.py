# monitoring-service/src/services/jaeger_service.py
"""Jaeger service management for distributed tracing"""

import asyncio
import httpx
import docker
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import statistics

from core.config import settings
from core.opentelemetry_setup import trace_function
from models.traces import (
    Trace, Span, Service, Operation, TraceSearchParams,
    TraceSearchResponse, ServiceDependencies, ServiceDependency,
    TraceTimeline, TraceMetrics, ServiceMetrics, TraceComparison,
    LatencyPercentiles, Flamegraph, FlamegraphNode, ServiceMap,
    ServiceNode, ServiceEdge, Process
)

logger = logging.getLogger(__name__)

class JaegerService:
    """Service for managing Jaeger distributed tracing"""
    
    def __init__(self):
        self.base_url = settings.JAEGER_URL
        self.query_url = settings.JAEGER_QUERY_URL
        self.api_url = f"{self.query_url}/api"
        self.client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for trace queries
        self.docker_client = docker.from_env()
        self.container_name = "jaeger"
        
    async def start(self):
        """Start Jaeger container if not already running"""
        try:
            # Check if container exists
            try:
                container = self.docker_client.containers.get(self.container_name)
                if container.status != "running":
                    logger.info(f"Starting existing {self.container_name} container")
                    container.start()
                    await self._wait_for_ready()
                else:
                    logger.info(f"{self.container_name} is already running")
            except docker.errors.NotFound:
                logger.info(f"Container {self.container_name} not found, will be created by docker-compose")
                await self._wait_for_ready()
                
        except Exception as e:
            logger.error(f"Failed to start Jaeger: {e}")
            raise
            
    async def stop(self):
        """Stop Jaeger gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop Jaeger: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for Jaeger to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/")
                if response.status_code == 200:
                    logger.info("Jaeger is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("Jaeger failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check Jaeger health"""
        try:
            # Check UI
            ui_response = await self.client.get(f"{self.base_url}/")
            ui_healthy = ui_response.status_code == 200
            
            # Check API
            api_response = await self.client.get(f"{self.api_url}/services")
            api_healthy = api_response.status_code == 200
            
            # Check health endpoint
            health_response = await self.client.get(f"{self.query_url}:14269/")
            admin_healthy = health_response.status_code == 200
            
            status = "healthy" if ui_healthy and api_healthy else "unhealthy"
            if ui_healthy and not api_healthy:
                status = "degraded"
            
            return {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "ui_healthy": ui_healthy,
                    "api_healthy": api_healthy,
                    "admin_healthy": admin_healthy
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": self.base_url}
            }
    
    @trace_function()
    async def search_traces(
        self,
        service: str,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration: Optional[str] = None,
        max_duration: Optional[str] = None,
        limit: int = 20
    ) -> TraceSearchResponse:
        """Search for traces based on criteria"""
        try:
            # Build search parameters
            params = TraceSearchParams(
                service=service,
                operation=operation,
                tags=tags,
                start_time=start_time,
                end_time=end_time,
                min_duration=min_duration,
                max_duration=max_duration,
                limit=limit
            )
            
            # Convert to query parameters
            query_params = params.to_query_params()
            
            # Search traces
            response = await self.client.get(
                f"{self.api_url}/traces",
                params=query_params
            )
            response.raise_for_status()
            
            data = response.json()
            traces = []
            
            for trace_data in data.get("data", []):
                trace = self._parse_trace(trace_data)
                traces.append(trace)
            
            return TraceSearchResponse(
                data=traces,
                total=len(traces),
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Failed to search traces: {e}")
            raise
    
    @trace_function()
    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a specific trace by ID"""
        try:
            response = await self.client.get(f"{self.api_url}/traces/{trace_id}")
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                return self._parse_trace(data["data"][0])
            
            return None
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            raise
    
    def _parse_trace(self, trace_data: Dict[str, Any]) -> Trace:
        """Parse trace data from Jaeger API"""
        spans = []
        for span_data in trace_data.get("spans", []):
            span = Span(**span_data)
            spans.append(span)
        
        processes = {}
        for proc_id, proc_data in trace_data.get("processes", {}).items():
            process = Process(**proc_data)
            processes[proc_id] = process
        
        return Trace(
            traceID=trace_data["traceID"],
            spans=spans,
            processes=processes,
            warnings=trace_data.get("warnings")
        )
    
    @trace_function()
    async def get_services(self) -> List[Service]:
        """Get all services that have reported traces"""
        try:
            response = await self.client.get(f"{self.api_url}/services")
            response.raise_for_status()
            
            data = response.json()
            services = []
            
            for service_name in data.get("data", []):
                service = Service(name=service_name)
                services.append(service)
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to get services: {e}")
            raise
    
    @trace_function()
    async def get_operations(self, service: str) -> List[Operation]:
        """Get all operations for a specific service"""
        try:
            response = await self.client.get(
                f"{self.api_url}/services/{service}/operations"
            )
            response.raise_for_status()
            
            data = response.json()
            operations = []
            
            for op_name in data.get("data", []):
                operation = Operation(name=op_name, service=service)
                operations.append(operation)
            
            return operations
            
        except Exception as e:
            logger.error(f"Failed to get operations for {service}: {e}")
            raise
    
    @trace_function()
    async def get_dependencies(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> ServiceDependencies:
        """Get service dependency graph"""
        try:
            params = {
                "endTs": int(end_time.timestamp() * 1000),
                "lookback": int((end_time - start_time).total_seconds() * 1000)
            }
            
            response = await self.client.get(
                f"{self.api_url}/dependencies",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            dependencies = []
            
            for dep in data.get("data", []):
                dependency = ServiceDependency(
                    parent=dep["parent"],
                    child=dep["child"],
                    callCount=dep.get("callCount", 0)
                )
                dependencies.append(dependency)
            
            return ServiceDependencies(
                dependencies=dependencies,
                errors=data.get("errors")
            )
            
        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            raise
    
    @trace_function()
    async def build_trace_timeline(self, trace: Union[Trace, Dict]) -> TraceTimeline:
        """Build timeline visualization data for a trace"""
        try:
            if isinstance(trace, dict):
                trace = self._parse_trace(trace)
            
            service_timelines = defaultdict(list)
            
            for span in trace.spans:
                process = trace.processes.get(span.processID, Process(serviceName="unknown", tags=[]))
                service_name = process.serviceName
                
                timeline_entry = {
                    "span_id": span.spanID,
                    "operation": span.operationName,
                    "start_ms": span.startTime / 1000,  # Convert to milliseconds
                    "duration_ms": span.duration / 1000,
                    "has_error": span.has_error(),
                    "tags": {tag.key: tag.value for tag in span.tags}
                }
                
                service_timelines[service_name].append(timeline_entry)
            
            # Find critical path (longest sequence)
            critical_path = self._find_critical_path(trace)
            
            return TraceTimeline(
                trace_id=trace.traceID,
                duration_ms=trace.duration_ms,
                service_timelines=dict(service_timelines),
                critical_path=critical_path
            )
            
        except Exception as e:
            logger.error(f"Failed to build trace timeline: {e}")
            raise
    
    def _find_critical_path(self, trace: Trace) -> List[str]:
        """Find the critical path in a trace"""
        # Build span relationships
        span_map = {span.spanID: span for span in trace.spans}
        children_map = defaultdict(list)
        
        for span in trace.spans:
            for ref in span.references:
                if ref.refType == "child_of":
                    children_map[ref.spanID].append(span.spanID)
        
        # Find root span
        root_span = trace.get_root_span()
        if not root_span:
            return []
        
        # DFS to find longest path
        def find_longest_path(span_id: str) -> Tuple[int, List[str]]:
            span = span_map.get(span_id)
            if not span:
                return 0, []
            
            if span_id not in children_map:
                return span.duration, [span_id]
            
            max_duration = 0
            max_path = []
            
            for child_id in children_map[span_id]:
                child_duration, child_path = find_longest_path(child_id)
                if child_duration > max_duration:
                    max_duration = child_duration
                    max_path = child_path
            
            return span.duration + max_duration, [span_id] + max_path
        
        _, critical_path = find_longest_path(root_span.spanID)
        return critical_path
    
    @trace_function()
    async def calculate_trace_metrics(self, trace: Union[Trace, Dict]) -> TraceMetrics:
        """Calculate metrics for a specific trace"""
        try:
            if isinstance(trace, dict):
                trace = self._parse_trace(trace)
            
            by_service = defaultdict(lambda: {
                "span_count": 0,
                "total_duration": 0,
                "error_count": 0,
                "operations": set()
            })
            
            by_operation = defaultdict(lambda: {
                "count": 0,
                "total_duration": 0,
                "error_count": 0,
                "services": set()
            })
            
            for span in trace.spans:
                process = trace.processes.get(span.processID, Process(serviceName="unknown", tags=[]))
                service_name = process.serviceName
                
                # By service metrics
                by_service[service_name]["span_count"] += 1
                by_service[service_name]["total_duration"] += span.duration_ms
                by_service[service_name]["operations"].add(span.operationName)
                if span.has_error():
                    by_service[service_name]["error_count"] += 1
                
                # By operation metrics
                by_operation[span.operationName]["count"] += 1
                by_operation[span.operationName]["total_duration"] += span.duration_ms
                by_operation[span.operationName]["services"].add(service_name)
                if span.has_error():
                    by_operation[span.operationName]["error_count"] += 1
            
            # Convert sets to counts
            for service_data in by_service.values():
                service_data["operation_count"] = len(service_data["operations"])
                del service_data["operations"]
            
            for op_data in by_operation.values():
                op_data["service_count"] = len(op_data["services"])
                del op_data["services"]
            
            return TraceMetrics(
                trace_id=trace.traceID,
                total_spans=trace.span_count,
                duration_ms=trace.duration_ms,
                services_involved=len(trace.service_names),
                error_count=trace.error_count,
                by_service=dict(by_service),
                by_operation=dict(by_operation)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate trace metrics: {e}")
            raise
    
    @trace_function()
    async def get_service_metrics(
        self,
        service: str,
        start_time: datetime,
        end_time: datetime
    ) -> ServiceMetrics:
        """Get aggregated metrics for a service"""
        try:
            # Search for traces for this service
            search_result = await self.search_traces(
                service=service,
                start_time=start_time,
                end_time=end_time,
                limit=1000  # Get more traces for better statistics
            )
            
            if not search_result.data:
                return ServiceMetrics(
                    service=service,
                    operation_count=0,
                    trace_count=0,
                    error_rate=0.0,
                    avg_duration_ms=0.0,
                    p50_duration_ms=0.0,
                    p95_duration_ms=0.0,
                    p99_duration_ms=0.0
                )
            
            # Calculate metrics
            operations = set()
            durations = []
            error_count = 0
            
            for trace in search_result.data:
                durations.append(trace.duration_ms)
                
                for span in trace.spans:
                    process = trace.processes.get(span.processID)
                    if process and process.serviceName == service:
                        operations.add(span.operationName)
                        if span.has_error():
                            error_count += 1
            
            # Calculate percentiles
            durations.sort()
            
            return ServiceMetrics(
                service=service,
                operation_count=len(operations),
                trace_count=len(search_result.data),
                error_rate=(error_count / len(search_result.data)) * 100 if search_result.data else 0,
                avg_duration_ms=statistics.mean(durations) if durations else 0,
                p50_duration_ms=self._percentile(durations, 0.50),
                p95_duration_ms=self._percentile(durations, 0.95),
                p99_duration_ms=self._percentile(durations, 0.99)
            )
            
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            raise
    
    def _percentile(self, sorted_list: List[float], percentile: float) -> float:
        """Calculate percentile from sorted list"""
        if not sorted_list:
            return 0.0
        
        index = int(len(sorted_list) * percentile)
        if index >= len(sorted_list):
            index = len(sorted_list) - 1
        
        return sorted_list[index]
    
    @trace_function()
    async def compare_traces(self, trace_ids: List[str]) -> TraceComparison:
        """Compare multiple traces"""
        try:
            traces = []
            for trace_id in trace_ids:
                trace = await self.get_trace(trace_id)
                if trace:
                    traces.append(trace)
            
            if len(traces) < 2:
                raise ValueError("Need at least 2 traces to compare")
            
            # Find common operations
            operation_sets = []
            for trace in traces:
                ops = set()
                for span in trace.spans:
                    ops.add(span.operationName)
                operation_sets.append(ops)
            
            common_operations = list(operation_sets[0].intersection(*operation_sets[1:]))
            
            # Compare durations
            duration_comparison = {
                trace.traceID: trace.duration_ms
                for trace in traces
            }
            
            # Compare span counts
            span_count_comparison = {
                trace.traceID: trace.span_count
                for trace in traces
            }
            
            # Compare services
            service_comparison = {
                trace.traceID: trace.service_names
                for trace in traces
            }
            
            # Compare operation timings
            operation_timings = defaultdict(dict)
            
            for trace in traces:
                op_durations = defaultdict(list)
                
                for span in trace.spans:
                    op_durations[span.operationName].append(span.duration_ms)
                
                for op, durations in op_durations.items():
                    if op in common_operations:
                        operation_timings[op][trace.traceID] = statistics.mean(durations)
            
            return TraceComparison(
                trace_ids=trace_ids,
                common_operations=common_operations,
                duration_comparison=duration_comparison,
                span_count_comparison=span_count_comparison,
                service_comparison=service_comparison,
                operation_timings=dict(operation_timings)
            )
            
        except Exception as e:
            logger.error(f"Failed to compare traces: {e}")
            raise
    
    @trace_function()
    async def get_latency_percentiles(
        self,
        service: str,
        operation: Optional[str] = None,
        percentiles: List[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> LatencyPercentiles:
        """Get latency percentiles for a service/operation"""
        try:
            if percentiles is None:
                percentiles = [0.5, 0.75, 0.95, 0.99]
            
            # Default time range
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=1)
            
            # Search for traces
            search_result = await self.search_traces(
                service=service,
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            # Collect latencies
            latencies = []
            
            for trace in search_result.data:
                if operation:
                    # Get latency for specific operation
                    for span in trace.spans:
                        process = trace.processes.get(span.processID)
                        if (process and process.serviceName == service and 
                            span.operationName == operation):
                            latencies.append(span.duration_ms)
                else:
                    # Get overall trace latency
                    latencies.append(trace.duration_ms)
            
            # Calculate percentiles
            latencies.sort()
            percentile_values = {}
            
            for p in percentiles:
                percentile_values[f"p{int(p*100)}"] = self._percentile(latencies, p)
            
            return LatencyPercentiles(
                service=service,
                operation=operation,
                percentiles=percentile_values,
                sample_count=len(latencies),
                time_range={
                    "start": start_time,
                    "end": end_time
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get latency percentiles: {e}")
            raise
    
    @trace_function()
    async def build_flamegraph(self, trace: Union[Trace, Dict]) -> Flamegraph:
        """Build flamegraph data for a trace"""
        try:
            if isinstance(trace, dict):
                trace = self._parse_trace(trace)
            
            # Build span relationships
            span_map = {span.spanID: span for span in trace.spans}
            children_map = defaultdict(list)
            
            for span in trace.spans:
                for ref in span.references:
                    if ref.refType == "child_of":
                        children_map[ref.spanID].append(span.spanID)
            
            # Find root span
            root_span = trace.get_root_span()
            if not root_span:
                raise ValueError("No root span found in trace")
            
            # Build flamegraph recursively
            def build_node(span_id: str) -> FlamegraphNode:
                span = span_map[span_id]
                process = trace.processes.get(span.processID)
                service_name = process.serviceName if process else "unknown"
                
                node = FlamegraphNode(
                    name=f"{service_name}.{span.operationName}",
                    value=span.duration,
                    children=[]
                )
                
                # Add children
                for child_id in children_map.get(span_id, []):
                    child_node = build_node(child_id)
                    node.children.append(child_node)
                
                return node
            
            root_node = build_node(root_span.spanID)
            
            return Flamegraph(
                trace_id=trace.traceID,
                root=root_node,
                total_time=trace.duration
            )
            
        except Exception as e:
            logger.error(f"Failed to build flamegraph: {e}")
            raise
    
    @trace_function()
    async def build_service_map(
        self,
        depth: int = 3,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ServiceMap:
        """Build service dependency map"""
        try:
            # Default time range
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(days=1)
            
            # Get dependencies
            dependencies = await self.get_dependencies(start_time, end_time)
            
            # Build nodes and edges
            nodes = []
            edges = []
            service_set = set()
            
            # Add all services as nodes
            for dep in dependencies.dependencies:
                service_set.add(dep.parent)
                service_set.add(dep.child)
            
            # Create nodes with metrics
            for service in service_set:
                try:
                    metrics = await self.get_service_metrics(service, start_time, end_time)
                    node = ServiceNode(
                        id=service,
                        label=service,
                        metrics=metrics
                    )
                except:
                    # If metrics fail, create node without metrics
                    node = ServiceNode(
                        id=service,
                        label=service
                    )
                nodes.append(node)
            
            # Create edges
            for dep in dependencies.dependencies:
                edge = ServiceEdge(
                    source=dep.parent,
                    target=dep.child,
                    call_count=dep.callCount
                )
                edges.append(edge)
            
            return ServiceMap(
                nodes=nodes,
                edges=edges,
                depth=depth,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to build service map: {e}")
            raise
    
    @trace_function()
    async def get_trace_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics for traces"""
        try:
            # Default time range
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            # Get services
            services = await self.get_services()
            
            stats = {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "services": {
                    "total": len(services),
                    "list": [s.name for s in services]
                },
                "traces": {},
                "operations": {},
                "errors": {}
            }
            
            # Get stats for each service or specific service
            services_to_check = [service] if service else [s.name for s in services]
            
            total_traces = 0
            total_errors = 0
            all_operations = set()
            
            for svc in services_to_check:
                try:
                    # Get service metrics
                    metrics = await self.get_service_metrics(svc, start_time, end_time)
                    
                    stats["traces"][svc] = {
                        "count": metrics.trace_count,
                        "error_rate": metrics.error_rate,
                        "avg_duration_ms": metrics.avg_duration_ms,
                        "p95_duration_ms": metrics.p95_duration_ms
                    }
                    
                    total_traces += metrics.trace_count
                    total_errors += int(metrics.trace_count * metrics.error_rate / 100)
                    
                    # Get operations
                    operations = await self.get_operations(svc)
                    for op in operations:
                        all_operations.add(op.name)
                    
                except Exception as e:
                    logger.warning(f"Failed to get stats for service {svc}: {e}")
            
            stats["summary"] = {
                "total_traces": total_traces,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_traces * 100) if total_traces > 0 else 0,
                "total_operations": len(all_operations)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get trace statistics: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
jaeger_service = JaegerService()