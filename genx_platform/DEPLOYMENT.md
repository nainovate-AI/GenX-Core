
# Production Deployment Guide

  

## Prerequisites

- Docker 24.0+

- Docker Compose 2.23+

- 16GB RAM minimum

- 100GB disk space

  

## Deployment Steps

  

1.  **Build Production Images**

	```bash
	make -f Makefile.prod build
	```

2.  **Run Security Scan**

	```bash
	make -f Makefile.prod security-scan
	```

3.  ***Deploy Services***

	```bash
	make -f Makefile.prod deploy
	```

4.  ***Verify Health***

	```bash
	make -f Makefile.prod health-check
	```

***Monitoring***

- Grafana: http://localhost:3000

- Prometheus: http://localhost:9090

- Jaeger: http://localhost:16686

  

***Scaling***

```bash
make -f Makefile.prod scale
```

  

***Backup***

```bash
make -f Makefile.prod backup
```

  

***Troubleshooting***

Check logs:

```bash
make -f Makefile.prod logs-llm
```

Access shell:

```bash
make -f Makefile.prod shell
```