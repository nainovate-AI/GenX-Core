# GenX Platform Monitoring Stack

This directory contains the platform-level monitoring and observability infrastructure for the GenX platform.

## Structure

- `prometheus/` - Metrics storage and querying
- `grafana/` - Metrics visualization and dashboards
- `jaeger/` - Distributed tracing
- `loki/` - Log aggregation
- `alertmanager/` - Alert routing and management
- `otel-collector/` - OpenTelemetry collector configuration
- `scripts/` - Monitoring maintenance scripts
- `docs/` - Monitoring documentation
- `certs/` - TLS certificates for secure communication

## Usage

The monitoring stack is deployed using Docker Compose:

```bash
# From genx_platform root
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

## Configuration

Each component has its own configuration directory. Modify the relevant config files and restart the service to apply changes.
