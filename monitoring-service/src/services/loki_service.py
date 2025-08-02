# monitoring-service/src/services/loki_service.py
"""Loki service management for log aggregation"""

import asyncio
import httpx
import docker
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import logging
from urllib.parse import urlencode
import websockets

from core.config import settings
from core.opentelemetry_setup import trace_function
from models.logs import (
    LogEntry, LogStream, LogQuery, LogQueryResponse, LogStreamResponse,
    LogLabelsResponse, LogLabelValuesResponse, LogSeriesResponse,
    LogMetricQuery, LogMetricResponse, LogStats, LogStreamStats,
    LogFilter, LogIngestionRequest, LogDirection
)

logger = logging.getLogger(__name__)

class LokiService:
    """Service for managing Loki log aggregation"""
    
    def __init__(self):
        self.base_url = settings.LOKI_URL
        self.push_url = settings.LOKI_PUSH_URL
        self.api_url = f"{self.base_url}/loki/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for log queries
        self.docker_client = docker.from_env()
        self.container_name = "loki"
        
    async def start(self):
        """Start Loki container if not already running"""
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
            logger.error(f"Failed to start Loki: {e}")
            raise
            
    async def stop(self):
        """Stop Loki gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop Loki: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for Loki to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/ready")
                if response.status_code == 200:
                    logger.info("Loki is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("Loki failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check Loki health"""
        try:
            # Check ready endpoint
            ready_response = await self.client.get(f"{self.base_url}/ready")
            ready = ready_response.status_code == 200
            
            # Check metrics endpoint
            metrics_response = await self.client.get(f"{self.base_url}/metrics")
            has_metrics = metrics_response.status_code == 200
            
            # Get build info
            build_info = {}
            try:
                build_response = await self.client.get(f"{self.base_url}/loki/api/v1/status/buildinfo")
                if build_response.status_code == 200:
                    build_info = build_response.json()
            except:
                pass
            
            return {
                "status": "healthy" if ready else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "ready": ready,
                    "metrics_available": has_metrics,
                    "version": build_info.get("version", "unknown"),
                    "build_time": build_info.get("buildTime", "unknown")
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
    async def query(self, query: str, time: Optional[datetime] = None) -> LogQueryResponse:
        """Execute an instant log query"""
        try:
            params = {"query": query}
            if time:
                params["time"] = str(int(time.timestamp() * 1e9))  # Nanoseconds
            
            response = await self.client.get(
                f"{self.api_url}/query",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return LogQueryResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to query logs: {e}")
            raise
    
    @trace_function()
    async def query_range(
        self,
        query: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        direction: LogDirection = LogDirection.BACKWARD,
        step: Optional[str] = None
    ) -> LogQueryResponse:
        """Execute a range log query"""
        try:
            # Default time range
            if not end:
                end = datetime.utcnow()
            if not start:
                start = end - timedelta(hours=1)
            
            params = {
                "query": query,
                "start": str(int(start.timestamp() * 1e9)),  # Nanoseconds
                "end": str(int(end.timestamp() * 1e9)),
                "limit": limit,
                "direction": direction.value
            }
            
            if step:
                params["step"] = step
            
            response = await self.client.get(
                f"{self.api_url}/query_range",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return LogQueryResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to query range logs: {e}")
            raise
    
    @trace_function()
    async def get_labels(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> LogLabelsResponse:
        """Get all label names"""
        try:
            params = {}
            if start:
                params["start"] = str(int(start.timestamp() * 1e9))
            if end:
                params["end"] = str(int(end.timestamp() * 1e9))
            
            response = await self.client.get(
                f"{self.api_url}/labels",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return LogLabelsResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to get labels: {e}")
            raise
    
    @trace_function()
    async def get_label_values(
        self,
        label: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> LogLabelValuesResponse:
        """Get all values for a specific label"""
        try:
            params = {}
            if start:
                params["start"] = str(int(start.timestamp() * 1e9))
            if end:
                params["end"] = str(int(end.timestamp() * 1e9))
            
            response = await self.client.get(
                f"{self.api_url}/label/{label}/values",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return LogLabelValuesResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to get label values for {label}: {e}")
            raise
    
    @trace_function()
    async def get_series(
        self,
        match: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> LogSeriesResponse:
        """Get series matching label matchers"""
        try:
            params = {"match": match}
            if start:
                params["start"] = str(int(start.timestamp() * 1e9))
            if end:
                params["end"] = str(int(end.timestamp() * 1e9))
            
            response = await self.client.get(
                f"{self.api_url}/series",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return LogSeriesResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to get series: {e}")
            raise
    
    @trace_function()
    async def push_logs(self, entries: List[LogEntry]) -> bool:
        """Push log entries to Loki"""
        try:
            # Group entries by labels
            streams = {}
            
            for entry in entries:
                # Create stream key from labels
                label_key = json.dumps(entry.labels, sort_keys=True)
                
                if label_key not in streams:
                    streams[label_key] = {
                        "stream": entry.labels,
                        "values": []
                    }
                
                # Add entry to stream
                timestamp_ns = str(int(entry.timestamp.timestamp() * 1e9))
                streams[label_key]["values"].append([timestamp_ns, entry.line])
            
            # Create push request
            push_data = {
                "streams": list(streams.values())
            }
            
            response = await self.client.post(
                self.push_url,
                json=push_data,
                headers={"Content-Type": "application/json"}
            )
            
            success = response.status_code in [200, 204]
            if success:
                logger.debug(f"Pushed {len(entries)} log entries to Loki")
            else:
                logger.warning(f"Failed to push logs: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to push logs to Loki: {e}")
            return False
    
    async def stream_logs(
        self,
        query: str,
        start: Optional[datetime] = None,
        delay_for: int = 0
    ) -> AsyncGenerator[LogEntry, None]:
        """Stream logs in real-time using WebSocket"""
        try:
            # Build WebSocket URL
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            
            params = {
                "query": query,
                "delay_for": delay_for
            }
            
            if start:
                params["start"] = str(int(start.timestamp() * 1e9))
            
            url = f"{ws_url}/loki/api/v1/tail?{urlencode(params)}"
            
            async with websockets.connect(url) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Parse streams
                    for stream_data in data.get("streams", []):
                        stream = LogStream(**stream_data)
                        for entry in stream.to_entries():
                            yield entry
                            
        except Exception as e:
            logger.error(f"Failed to stream logs: {e}")
            raise
    
    @trace_function()
    async def get_stats(self) -> LogStats:
        """Get Loki statistics"""
        try:
            # Get metrics from Loki
            response = await self.client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            
            metrics_text = response.text
            
            # Parse relevant metrics (simplified parsing)
            stats = LogStats(
                ingested_lines=self._parse_metric(metrics_text, "loki_ingester_streams_created_total"),
                ingested_bytes=self._parse_metric(metrics_text, "loki_ingester_bytes_received_total"),
                compressed_bytes=self._parse_metric(metrics_text, "loki_ingester_chunk_stored_bytes_total"),
                streams_count=self._parse_metric(metrics_text, "loki_ingester_memory_streams"),
                chunks_count=self._parse_metric(metrics_text, "loki_ingester_memory_chunks"),
                entries_per_second=self._parse_metric(metrics_text, "loki_ingester_entries_received_total"),
                bytes_per_second=self._parse_metric(metrics_text, "loki_ingester_bytes_received_total")
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get Loki stats: {e}")
            # Return default stats on error
            return LogStats(
                ingested_lines=0,
                ingested_bytes=0,
                compressed_bytes=0,
                streams_count=0,
                chunks_count=0,
                entries_per_second=0.0,
                bytes_per_second=0.0
            )
    
    def _parse_metric(self, metrics_text: str, metric_name: str) -> float:
        """Parse a metric value from Prometheus format text"""
        try:
            for line in metrics_text.split('\n'):
                if line.startswith(metric_name):
                    # Extract value after the metric name
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[-1])
            return 0.0
        except:
            return 0.0
    
    @trace_function()
    async def tail_logs(
        self,
        query: str,
        lines: int = 100,
        start: Optional[datetime] = None
    ) -> List[LogEntry]:
        """Tail recent logs"""
        try:
            # Query recent logs
            if not start:
                start = datetime.utcnow() - timedelta(minutes=5)
            
            result = await self.query_range(
                query=query,
                start=start,
                end=datetime.utcnow(),
                limit=lines,
                direction=LogDirection.BACKWARD
            )
            
            return result.entries
            
        except Exception as e:
            logger.error(f"Failed to tail logs: {e}")
            raise
    
    @trace_function()
    async def get_log_context(
        self,
        query: str,
        timestamp: datetime,
        before: int = 10,
        after: int = 10
    ) -> Dict[str, Any]:
        """Get log context (lines before and after a specific log)"""
        try:
            # Query for logs around the timestamp
            time_buffer = timedelta(minutes=5)
            
            # Get logs before
            before_result = await self.query_range(
                query=query,
                start=timestamp - time_buffer,
                end=timestamp,
                limit=before + 1,  # +1 to include target
                direction=LogDirection.BACKWARD
            )
            
            # Get logs after
            after_result = await self.query_range(
                query=query,
                start=timestamp,
                end=timestamp + time_buffer,
                limit=after + 1,  # +1 to include target
                direction=LogDirection.FORWARD
            )
            
            # Find target entry
            target_entry = None
            all_entries = before_result.entries + after_result.entries
            
            for entry in all_entries:
                if abs((entry.timestamp - timestamp).total_seconds()) < 1:
                    target_entry = entry
                    break
            
            # Extract context entries
            before_entries = []
            after_entries = []
            
            if target_entry:
                for entry in before_result.entries:
                    if entry.timestamp < target_entry.timestamp:
                        before_entries.append(entry)
                
                for entry in after_result.entries:
                    if entry.timestamp > target_entry.timestamp:
                        after_entries.append(entry)
            
            return {
                "target_entry": target_entry,
                "before_entries": before_entries[-before:],  # Last N entries
                "after_entries": after_entries[:after]  # First N entries
            }
            
        except Exception as e:
            logger.error(f"Failed to get log context: {e}")
            raise
    
    @trace_function()
    async def aggregate_logs(
        self,
        query: str,
        group_by: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate logs by labels"""
        try:
            # Default time range
            if not end:
                end = datetime.utcnow()
            if not start:
                start = end - timedelta(hours=1)
            
            # Build aggregation query
            group_expr = ",".join(group_by)
            agg_query = f'sum by ({group_expr}) (count_over_time({query}[5m]))'
            
            # Execute metric query
            response = await self.client.get(
                f"{self.api_url}/query_range",
                params={
                    "query": agg_query,
                    "start": str(int(start.timestamp())),
                    "end": str(int(end.timestamp())),
                    "step": "5m"
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Process results
            aggregations = []
            for result in data.get("data", {}).get("result", []):
                aggregations.append({
                    "labels": result["metric"],
                    "values": result["values"],
                    "total": sum(float(v[1]) for v in result["values"])
                })
            
            # Sort by total descending
            aggregations.sort(key=lambda x: x["total"], reverse=True)
            
            return aggregations
            
        except Exception as e:
            logger.error(f"Failed to aggregate logs: {e}")
            raise
    
    @trace_function()
    async def search_logs(
        self,
        filter: LogFilter,
        limit: int = 100
    ) -> List[LogEntry]:
        """Search logs using a filter"""
        try:
            # Convert filter to LogQL query
            query = filter.to_logql()
            
            # Execute query
            result = await self.query_range(
                query=query,
                start=filter.start_time,
                end=filter.end_time,
                limit=limit
            )
            
            return result.entries
            
        except Exception as e:
            logger.error(f"Failed to search logs: {e}")
            raise
    
    @trace_function()
    async def get_log_patterns(
        self,
        service: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        sample_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Analyze logs to find common patterns"""
        try:
            # Default time range
            if not end:
                end = datetime.utcnow()
            if not start:
                start = end - timedelta(hours=1)
            
            # Query logs for the service
            query = f'{{service="{service}"}}'
            result = await self.query_range(
                query=query,
                start=start,
                end=end,
                limit=sample_size
            )
            
            # Simple pattern detection (in production, use more sophisticated methods)
            patterns = {}
            
            for entry in result.entries:
                # Remove timestamps, IDs, and numbers to find patterns
                import re
                pattern = entry.line
                pattern = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', pattern)
                pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', pattern)
                pattern = re.sub(r'\b\d+\b', 'NUM', pattern)
                pattern = re.sub(r'\b[a-f0-9]{32,}\b', 'HASH', pattern)
                
                if pattern in patterns:
                    patterns[pattern]["count"] += 1
                    patterns[pattern]["examples"].append(entry.line)
                else:
                    patterns[pattern] = {
                        "pattern": pattern,
                        "count": 1,
                        "examples": [entry.line]
                    }
            
            # Convert to list and sort by frequency
            pattern_list = list(patterns.values())
            pattern_list.sort(key=lambda x: x["count"], reverse=True)
            
            # Keep only top patterns and limit examples
            top_patterns = []
            for p in pattern_list[:20]:  # Top 20 patterns
                p["examples"] = p["examples"][:3]  # Keep 3 examples
                p["percentage"] = (p["count"] / len(result.entries)) * 100
                top_patterns.append(p)
            
            return top_patterns
            
        except Exception as e:
            logger.error(f"Failed to get log patterns: {e}")
            raise
    
    @trace_function()
    async def get_error_logs(
        self,
        service: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Get error logs"""
        try:
            # Build query for error logs
            if service:
                query = f'{{service="{service}"}} |~ "(?i)(error|exception|fail|critical)"'
            else:
                query = '{job=~".+"} |~ "(?i)(error|exception|fail|critical)"'
            
            # Default time range
            if not end:
                end = datetime.utcnow()
            if not start:
                start = end - timedelta(hours=1)
            
            result = await self.query_range(
                query=query,
                start=start,
                end=end,
                limit=limit,
                direction=LogDirection.BACKWARD
            )
            
            return result.entries
            
        except Exception as e:
            logger.error(f"Failed to get error logs: {e}")
            raise
    
    @trace_function()
    async def export_logs(
        self,
        query: str,
        start: datetime,
        end: datetime,
        format: str = "json",
        limit: int = 10000
    ) -> str:
        """Export logs in specified format"""
        try:
            # Query logs
            result = await self.query_range(
                query=query,
                start=start,
                end=end,
                limit=limit
            )
            
            if format == "json":
                # Export as JSON
                export_data = {
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "entries": [
                        {
                            "timestamp": entry.timestamp.isoformat(),
                            "line": entry.line,
                            "labels": entry.labels
                        }
                        for entry in result.entries
                    ]
                }
                return json.dumps(export_data, indent=2)
                
            elif format == "csv":
                # Export as CSV
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(["timestamp", "service", "level", "message"])
                
                # Write entries
                for entry in result.entries:
                    writer.writerow([
                        entry.timestamp.isoformat(),
                        entry.labels.get("service", ""),
                        entry.labels.get("level", ""),
                        entry.line
                    ])
                
                return output.getvalue()
                
            elif format == "txt":
                # Export as plain text
                lines = []
                for entry in result.entries:
                    lines.append(f"{entry.timestamp.isoformat()} {entry.line}")
                return "\n".join(lines)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
loki_service = LokiService()