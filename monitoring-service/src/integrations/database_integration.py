# monitoring-service/src/integrations/database_integration.py
"""Database integration for monitoring service"""

import time
import logging
import asyncio
from typing import Optional, Dict, Any, Callable, Union, List
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import sqlalchemy
from sqlalchemy import event, Engine
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool
import asyncpg
import motor.motor_asyncio
import redis.asyncio as redis
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.semconv.trace import SpanAttributes

from core.opentelemetry_setup import get_tracer, get_meter
from core.config import settings

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    PROMETHEUS = "prometheus"
    LOKI = "loki"
    JAEGER = "jaeger"


@dataclass
class QueryMetrics:
    """Query metrics data"""
    query_type: str
    duration: float
    rows_affected: int
    success: bool
    database: str
    table: Optional[str] = None
    error: Optional[str] = None


class DatabaseMonitor:
    """Base class for database monitoring"""
    
    def __init__(self, service_name: str, database_type: DatabaseType):
        self.service_name = service_name
        self.database_type = database_type
        self.tracer = get_tracer(f"{service_name}.db.{database_type.value}")
        self.meter = get_meter(f"{service_name}.db.{database_type.value}")
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup database metrics"""
        self.query_counter = self.meter.create_counter(
            name=f"db_{self.database_type.value}_queries_total",
            description=f"Total number of {self.database_type.value} queries",
            unit="1"
        )
        
        self.query_duration = self.meter.create_histogram(
            name=f"db_{self.database_type.value}_query_duration_seconds",
            description=f"{self.database_type.value} query duration in seconds",
            unit="s"
        )
        
        self.connection_counter = self.meter.create_up_down_counter(
            name=f"db_{self.database_type.value}_connections_active",
            description=f"Number of active {self.database_type.value} connections",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name=f"db_{self.database_type.value}_errors_total",
            description=f"Total number of {self.database_type.value} errors",
            unit="1"
        )
        
        self.pool_size = self.meter.create_up_down_counter(
            name=f"db_{self.database_type.value}_pool_size",
            description=f"{self.database_type.value} connection pool size",
            unit="1"
        )
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        labels = {
            "query_type": metrics.query_type,
            "database": metrics.database,
            "success": str(metrics.success)
        }
        
        if metrics.table:
            labels["table"] = metrics.table
        
        self.query_counter.add(1, labels)
        self.query_duration.record(metrics.duration, labels)
        
        if not metrics.success and metrics.error:
            self.error_counter.add(1, {
                "query_type": metrics.query_type,
                "error_type": type(metrics.error).__name__ if isinstance(metrics.error, Exception) else str(metrics.error)
            })


class SQLAlchemyMonitor(DatabaseMonitor):
    """Monitor for SQLAlchemy database operations"""
    
    def __init__(self, engine: Engine, service_name: str = None):
        db_type = self._get_database_type(engine)
        super().__init__(
            service_name or settings.SERVICE_NAME,
            db_type
        )
        self.engine = engine
        self._setup_event_listeners()
    
    def _get_database_type(self, engine: Engine) -> DatabaseType:
        """Detect database type from engine"""
        dialect = engine.dialect.name.lower()
        if "postgres" in dialect:
            return DatabaseType.POSTGRESQL
        elif "mysql" in dialect:
            return DatabaseType.MYSQL
        elif "sqlite" in dialect:
            return DatabaseType.SQLITE
        else:
            return DatabaseType.SQLITE  # Default
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners"""
        # Connection pool events
        event.listen(self.engine.pool, "connect", self._on_connect)
        event.listen(self.engine.pool, "checkout", self._on_checkout)
        event.listen(self.engine.pool, "checkin", self._on_checkin)
        
        # Query execution events
        event.listen(self.engine, "before_execute", self._before_execute)
        event.listen(self.engine, "after_execute", self._after_execute)
        event.listen(self.engine, "handle_error", self._handle_error)
    
    def _on_connect(self, dbapi_conn, connection_record):
        """Handle new connection creation"""
        self.pool_size.add(1)
        logger.debug(f"New {self.database_type.value} connection created")
    
    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Handle connection checkout from pool"""
        self.connection_counter.add(1)
    
    def _on_checkin(self, dbapi_conn, connection_record):
        """Handle connection checkin to pool"""
        self.connection_counter.add(-1)
    
    def _before_execute(self, conn: Connection, clauseelement, multiparams, params, execution_options):
        """Before query execution"""
        # Store start time in connection info
        conn.info["query_start_time"] = time.time()
        conn.info["query_span"] = self.tracer.start_as_current_span(
            self._get_operation_name(clauseelement),
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.DB_SYSTEM: self.database_type.value,
                SpanAttributes.DB_NAME: conn.engine.url.database,
                SpanAttributes.DB_STATEMENT: str(clauseelement)[:1000],  # Truncate long queries
                "db.table": self._extract_table_name(clauseelement)
            }
        ).__enter__()
    
    def _after_execute(self, conn: Connection, clauseelement, multiparams, params, execution_options, result):
        """After query execution"""
        if "query_start_time" in conn.info:
            duration = time.time() - conn.info["query_start_time"]
            
            # End span
            if "query_span" in conn.info:
                span = conn.info["query_span"]
                span.set_attribute("db.rows_affected", result.rowcount if hasattr(result, "rowcount") else 0)
                span.set_status(Status(StatusCode.OK))
                span.__exit__(None, None, None)
            
            # Record metrics
            self.record_query(QueryMetrics(
                query_type=self._get_query_type(clauseelement),
                duration=duration,
                rows_affected=result.rowcount if hasattr(result, "rowcount") else 0,
                success=True,
                database=conn.engine.url.database or "default",
                table=self._extract_table_name(clauseelement)
            ))
            
            # Clean up connection info
            conn.info.pop("query_start_time", None)
            conn.info.pop("query_span", None)
    
    def _handle_error(self, exception_context):
        """Handle query execution error"""
        conn = exception_context.connection
        
        if conn and "query_start_time" in conn.info:
            duration = time.time() - conn.info["query_start_time"]
            
            # End span with error
            if "query_span" in conn.info:
                span = conn.info["query_span"]
                span.record_exception(exception_context.original_exception)
                span.set_status(Status(StatusCode.ERROR, str(exception_context.original_exception)))
                span.__exit__(None, None, None)
            
            # Record error metrics
            self.record_query(QueryMetrics(
                query_type=self._get_query_type(exception_context.statement),
                duration=duration,
                rows_affected=0,
                success=False,
                database=conn.engine.url.database or "default",
                table=self._extract_table_name(exception_context.statement),
                error=str(exception_context.original_exception)
            ))
    
    def _get_operation_name(self, statement) -> str:
        """Get operation name from SQL statement"""
        query_type = self._get_query_type(statement)
        table = self._extract_table_name(statement)
        if table:
            return f"db.{query_type.lower()}.{table}"
        return f"db.{query_type.lower()}"
    
    def _get_query_type(self, statement) -> str:
        """Extract query type from statement"""
        if hasattr(statement, "is_select") and statement.is_select:
            return "SELECT"
        elif hasattr(statement, "is_insert") and statement.is_insert:
            return "INSERT"
        elif hasattr(statement, "is_update") and statement.is_update:
            return "UPDATE"
        elif hasattr(statement, "is_delete") and statement.is_delete:
            return "DELETE"
        else:
            # Parse from string
            sql_str = str(statement).strip().upper()
            for query_type in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]:
                if sql_str.startswith(query_type):
                    return query_type
            return "OTHER"
    
    def _extract_table_name(self, statement) -> Optional[str]:
        """Extract table name from statement"""
        try:
            if hasattr(statement, "table") and hasattr(statement.table, "name"):
                return statement.table.name
            elif hasattr(statement, "froms"):
                for table in statement.froms:
                    if hasattr(table, "name"):
                        return table.name
            # Try to parse from string
            sql_str = str(statement).upper()
            if "FROM" in sql_str:
                parts = sql_str.split("FROM")[1].strip().split()
                if parts:
                    return parts[0].strip('"').strip("'").lower()
        except:
            pass
        return None


class AsyncPGMonitor(DatabaseMonitor):
    """Monitor for AsyncPG PostgreSQL operations"""
    
    def __init__(self, service_name: str = None):
        super().__init__(
            service_name or settings.SERVICE_NAME,
            DatabaseType.POSTGRESQL
        )
    
    @asynccontextmanager
    async def monitor_query(
        self,
        query: str,
        database: str = "default",
        table: Optional[str] = None
    ):
        """Context manager for monitoring async queries"""
        query_type = self._get_query_type(query)
        
        with self.tracer.start_as_current_span(
            f"db.{query_type.lower()}.{table or 'unknown'}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.DB_SYSTEM: "postgresql",
                SpanAttributes.DB_STATEMENT: query[:1000],
                SpanAttributes.DB_NAME: database,
                "db.table": table
            }
        ) as span:
            start_time = time.time()
            
            try:
                yield span
                
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.OK))
                
                self.record_query(QueryMetrics(
                    query_type=query_type,
                    duration=duration,
                    rows_affected=0,  # Will be set by caller
                    success=True,
                    database=database,
                    table=table
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                
                self.record_query(QueryMetrics(
                    query_type=query_type,
                    duration=duration,
                    rows_affected=0,
                    success=False,
                    database=database,
                    table=table,
                    error=str(e)
                ))
                raise
    
    def _get_query_type(self, query: str) -> str:
        """Extract query type from SQL string"""
        query_upper = query.strip().upper()
        for query_type in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]:
            if query_upper.startswith(query_type):
                return query_type
        return "OTHER"


class MongoDBMonitor(DatabaseMonitor):
    """Monitor for MongoDB operations"""
    
    def __init__(self, service_name: str = None):
        super().__init__(
            service_name or settings.SERVICE_NAME,
            DatabaseType.MONGODB
        )
    
    @asynccontextmanager
    async def monitor_operation(
        self,
        operation: str,
        collection: str,
        database: str = "default"
    ):
        """Context manager for monitoring MongoDB operations"""
        with self.tracer.start_as_current_span(
            f"db.mongodb.{operation}.{collection}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.DB_SYSTEM: "mongodb",
                SpanAttributes.DB_OPERATION: operation,
                SpanAttributes.DB_NAME: database,
                "db.collection": collection
            }
        ) as span:
            start_time = time.time()
            
            try:
                yield span
                
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.OK))
                
                self.record_query(QueryMetrics(
                    query_type=operation.upper(),
                    duration=duration,
                    rows_affected=0,  # Will be set by caller
                    success=True,
                    database=database,
                    table=collection
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                
                self.record_query(QueryMetrics(
                    query_type=operation.upper(),
                    duration=duration,
                    rows_affected=0,
                    success=False,
                    database=database,
                    table=collection,
                    error=str(e)
                ))
                raise


class RedisMonitor(DatabaseMonitor):
    """Monitor for Redis operations"""
    
    def __init__(self, service_name: str = None):
        super().__init__(
            service_name or settings.SERVICE_NAME,
            DatabaseType.REDIS
        )
    
    @asynccontextmanager
    async def monitor_command(
        self,
        command: str,
        key: Optional[str] = None
    ):
        """Context manager for monitoring Redis commands"""
        with self.tracer.start_as_current_span(
            f"db.redis.{command.lower()}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.DB_SYSTEM: "redis",
                SpanAttributes.DB_OPERATION: command,
                "db.redis.key": key if key else "unknown"
            }
        ) as span:
            start_time = time.time()
            
            try:
                yield span
                
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.OK))
                
                self.record_query(QueryMetrics(
                    query_type=command.upper(),
                    duration=duration,
                    rows_affected=1,  # Redis typically affects one key
                    success=True,
                    database="redis"
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                
                self.record_query(QueryMetrics(
                    query_type=command.upper(),
                    duration=duration,
                    rows_affected=0,
                    success=False,
                    database="redis",
                    error=str(e)
                ))
                raise


# Decorators for database operations
def monitor_db_operation(
    operation_name: Optional[str] = None,
    database_type: DatabaseType = DatabaseType.POSTGRESQL
):
    """Decorator to monitor database operations"""
    def decorator(func):
        name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = DatabaseMonitor(settings.SERVICE_NAME, database_type)
            tracer = monitor.tracer
            
            with tracer.start_as_current_span(
                f"db.operation.{name}",
                kind=trace.SpanKind.CLIENT
            ) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    span.set_attribute("db.operation.duration", duration)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("db.operation.duration", duration)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    
                    monitor.error_counter.add(1, {
                        "operation": name,
                        "error_type": type(e).__name__
                    })
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = DatabaseMonitor(settings.SERVICE_NAME, database_type)
            tracer = monitor.tracer
            
            with tracer.start_as_current_span(
                f"db.operation.{name}",
                kind=trace.SpanKind.CLIENT
            ) as span:
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    span.set_attribute("db.operation.duration", duration)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("db.operation.duration", duration)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    
                    monitor.error_counter.add(1, {
                        "operation": name,
                        "error_type": type(e).__name__
                    })
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Helper classes for specific database integrations
class MonitoredAsyncPGPool:
    """Monitored AsyncPG connection pool"""
    
    def __init__(self, pool: asyncpg.Pool, monitor: AsyncPGMonitor):
        self.pool = pool
        self.monitor = monitor
    
    async def execute(self, query: str, *args, **kwargs):
        """Execute query with monitoring"""
        async with self.monitor.monitor_query(query):
            return await self.pool.execute(query, *args, **kwargs)
    
    async def fetch(self, query: str, *args, **kwargs):
        """Fetch rows with monitoring"""
        async with self.monitor.monitor_query(query):
            return await self.pool.fetch(query, *args, **kwargs)
    
    async def fetchrow(self, query: str, *args, **kwargs):
        """Fetch single row with monitoring"""
        async with self.monitor.monitor_query(query):
            return await self.pool.fetchrow(query, *args, **kwargs)
    
    async def fetchval(self, query: str, *args, **kwargs):
        """Fetch single value with monitoring"""
        async with self.monitor.monitor_query(query):
            return await self.pool.fetchval(query, *args, **kwargs)


class MonitoredMongoClient:
    """Monitored MongoDB client wrapper"""
    
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, monitor: MongoDBMonitor):
        self.client = client
        self.monitor = monitor
    
    def __getattr__(self, name):
        """Proxy attribute access to client"""
        return getattr(self.client, name)
    
    def get_database(self, name: str):
        """Get monitored database"""
        return MonitoredMongoDatabase(self.client[name], self.monitor, name)


class MonitoredMongoDatabase:
    """Monitored MongoDB database wrapper"""
    
    def __init__(self, database, monitor: MongoDBMonitor, db_name: str):
        self.database = database
        self.monitor = monitor
        self.db_name = db_name
    
    def __getattr__(self, name):
        """Proxy attribute access to database"""
        return getattr(self.database, name)
    
    def get_collection(self, name: str):
        """Get monitored collection"""
        return MonitoredMongoCollection(self.database[name], self.monitor, self.db_name, name)


class MonitoredMongoCollection:
    """Monitored MongoDB collection wrapper"""
    
    def __init__(self, collection, monitor: MongoDBMonitor, db_name: str, collection_name: str):
        self.collection = collection
        self.monitor = monitor
        self.db_name = db_name
        self.collection_name = collection_name
    
    async def insert_one(self, document: Dict, *args, **kwargs):
        """Insert one document with monitoring"""
        async with self.monitor.monitor_operation("insert_one", self.collection_name, self.db_name) as span:
            result = await self.collection.insert_one(document, *args, **kwargs)
            span.set_attribute("db.documents_affected", 1)
            return result
    
    async def insert_many(self, documents: List[Dict], *args, **kwargs):
        """Insert many documents with monitoring"""
        async with self.monitor.monitor_operation("insert_many", self.collection_name, self.db_name) as span:
            result = await self.collection.insert_many(documents, *args, **kwargs)
            span.set_attribute("db.documents_affected", len(result.inserted_ids))
            return result
    
    async def find_one(self, filter: Dict, *args, **kwargs):
        """Find one document with monitoring"""
        async with self.monitor.monitor_operation("find_one", self.collection_name, self.db_name) as span:
            span.set_attribute("db.filter", str(filter))
            return await self.collection.find_one(filter, *args, **kwargs)
    
    async def find(self, filter: Dict, *args, **kwargs):
        """Find documents with monitoring"""
        async with self.monitor.monitor_operation("find", self.collection_name, self.db_name) as span:
            span.set_attribute("db.filter", str(filter))
            return self.collection.find(filter, *args, **kwargs)
    
    async def update_one(self, filter: Dict, update: Dict, *args, **kwargs):
        """Update one document with monitoring"""
        async with self.monitor.monitor_operation("update_one", self.collection_name, self.db_name) as span:
            result = await self.collection.update_one(filter, update, *args, **kwargs)
            span.set_attribute("db.documents_affected", result.modified_count)
            return result
    
    async def update_many(self, filter: Dict, update: Dict, *args, **kwargs):
        """Update many documents with monitoring"""
        async with self.monitor.monitor_operation("update_many", self.collection_name, self.db_name) as span:
            result = await self.collection.update_many(filter, update, *args, **kwargs)
            span.set_attribute("db.documents_affected", result.modified_count)
            return result
    
    async def delete_one(self, filter: Dict, *args, **kwargs):
        """Delete one document with monitoring"""
        async with self.monitor.monitor_operation("delete_one", self.collection_name, self.db_name) as span:
            result = await self.collection.delete_one(filter, *args, **kwargs)
            span.set_attribute("db.documents_affected", result.deleted_count)
            return result
    
    async def delete_many(self, filter: Dict, *args, **kwargs):
        """Delete many documents with monitoring"""
        async with self.monitor.monitor_operation("delete_many", self.collection_name, self.db_name) as span:
            result = await self.collection.delete_many(filter, *args, **kwargs)
            span.set_attribute("db.documents_affected", result.deleted_count)
            return result
    
    def __getattr__(self, name):
        """Proxy other methods to collection"""
        return getattr(self.collection, name)


# Factory functions for creating monitored database clients
def create_monitored_engine(
    database_url: str,
    service_name: Optional[str] = None,
    **engine_kwargs
) -> Engine:
    """Create SQLAlchemy engine with monitoring"""
    engine = sqlalchemy.create_engine(database_url, **engine_kwargs)
    monitor = SQLAlchemyMonitor(engine, service_name)
    return engine


async def create_monitored_asyncpg_pool(
    dsn: str,
    service_name: Optional[str] = None,
    **pool_kwargs
) -> MonitoredAsyncPGPool:
    """Create AsyncPG pool with monitoring"""
    pool = await asyncpg.create_pool(dsn, **pool_kwargs)
    monitor = AsyncPGMonitor(service_name)
    return MonitoredAsyncPGPool(pool, monitor)


def create_monitored_mongo_client(
    connection_string: str,
    service_name: Optional[str] = None,
    **client_kwargs
) -> MonitoredMongoClient:
    """Create MongoDB client with monitoring"""
    client = motor.motor_asyncio.AsyncIOMotorClient(connection_string, **client_kwargs)
    monitor = MongoDBMonitor(service_name)
    return MonitoredMongoClient(client, monitor)