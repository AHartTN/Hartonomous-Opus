# MCP Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying MCP server and client functionality in the Hartonomous-Opus system across development, staging, and production environments.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), Windows Server 2019+, or macOS 12+
- **Runtime**: .NET 8.0 Runtime or SDK
- **Database**: PostgreSQL 15+ with PostGIS extension
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 10GB+ available disk space
- **Network**: Stable internet connection for external MCP clients

### Software Dependencies
```bash
# Install .NET 8.0 SDK
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0

# Install PostgreSQL and PostGIS
sudo apt-get install -y postgresql-15 postgresql-15-postgis-3

# Install Docker (optional, for containerized deployment)
sudo apt-get install -y docker.io docker-compose
```

### Network Configuration
- **Inbound Ports**:
  - 80/443: HTTP/HTTPS for REST API and MCP
  - 5001: MCP-specific port (configurable)
- **Outbound**: Allow connections to external MCP servers
- **SSL/TLS**: Required for production deployments
- **Firewalls**: Configure rules for MCP traffic

## Environment Configuration

### Environment Variables
```bash
# Application Configuration
export ASPNETCORE_ENVIRONMENT=Production
export ASPNETCORE_URLS="http://+:80;https://+:443"
export DOTNET_CLI_TELEMETRY_OPTOUT=1

# Database Configuration
export ConnectionStrings__HypercubeDatabase="Host=localhost;Port=5432;Database=hypercube;Username=hypercube_user;Password=${DB_PASSWORD};SSL Mode=Require;Trust Server Certificate=false"

# MCP Server Configuration
export MCP__Server__Enabled=true
export MCP__Server__Transport__Type=http
export MCP__Server__Transport__Path=/mcp
export MCP__Server__Transport__Port=5001

# MCP Security Configuration
export MCP__Security__RequireAuthentication=true
export MCP__Security__GlobalRateLimit__RequestsPerMinute=1000
export MCP__Security__GlobalRateLimit__RequestsPerHour=10000

# MCP Client Configuration
export MCP__Clients__ExternalTools__Transport=http
export MCP__Clients__ExternalTools__BaseUrl="https://api.external-mcp.com/mcp"
export MCP__Clients__ExternalTools__Headers__Authorization="Bearer ${EXTERNAL_API_TOKEN}"

# Authentication Configuration
export JWT__Issuer="hypercube-api"
export JWT__Audience="hypercube-clients"
export JWT__Key="${JWT_SECRET_KEY}"
export JWT__ExpiryMinutes=60

# Logging Configuration
export Logging__LogLevel__Default=Information
export Logging__LogLevel__Microsoft=Warning
export Logging__LogLevel__HypercubeGenerativeApi=Information
export Logging__LogLevel__MCP=Debug

# Monitoring Configuration
export OTEL__SERVICE__NAME="hypercube-mcp-api"
export OTEL__SERVICE__VERSION="1.0.0"
export OTEL__TRACES__EXPORTER=otlp
export OTEL__METRICS__EXPORTER=otlp
export OTEL__LOGS__EXPORTER=otlp
export OTEL__EXPORTER__OTLP__ENDPOINT="http://otel-collector:4317"
export OTEL__EXPORTER__OTLP__HEADERS="authorization=Bearer ${OTEL_TOKEN}"

# Health Check Configuration
export HealthChecks__UI__Enabled=true
export HealthChecks__UI__Port=5002
```

### Configuration Files

#### appsettings.Production.json
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning",
      "HypercubeGenerativeApi": "Information",
      "MCP": "Information"
    },
    "Console": {
      "FormatterName": "json"
    },
    "File": {
      "Path": "/var/log/hypercube/hypercube-.log",
      "RollingInterval": "Day",
      "RollingFileCount": 7
    }
  },
  "Mcp": {
    "Server": {
      "Enabled": true,
      "Transport": {
        "Type": "http",
        "Path": "/mcp",
        "Port": 5001
      },
      "Tools": {
        "SemanticSearch": {
          "Enabled": true,
          "RequiredPermissions": ["semantic:read"],
          "RateLimit": {
            "RequestsPerMinute": 100
          }
        },
        "GenerateText": {
          "Enabled": true,
          "RequiredPermissions": ["generation:execute"],
          "RateLimit": {
            "RequestsPerMinute": 10
          }
        },
        "GeometricNeighbors": {
          "Enabled": true,
          "RequiredPermissions": ["geometry:read"],
          "RateLimit": {
            "RequestsPerMinute": 50
          }
        },
        "FindAnalogies": {
          "Enabled": true,
          "RequiredPermissions": ["semantic:read"],
          "RateLimit": {
            "RequestsPerMinute": 20
          }
        },
        "IngestContent": {
          "Enabled": true,
          "RequiredPermissions": ["content:write", "semantic:write"],
          "RateLimit": {
            "RequestsPerMinute": 5
          }
        }
      }
    },
    "Clients": {
      "externalTools": {
        "Transport": "http",
        "BaseUrl": "https://api.external-tools.com/mcp",
        "Headers": {
          "Authorization": "Bearer ${EXTERNAL_TOOLS_TOKEN}",
          "User-Agent": "Hartonomous-Opus/1.0.0"
        },
        "Timeout": "00:00:30",
        "RetryPolicy": {
          "MaxRetries": 3,
          "BackoffMultiplier": 2.0
        }
      },
      "codeAnalysis": {
        "Transport": "stdio",
        "Command": "node",
        "Args": ["/opt/mcp-tools/code-analyzer/index.js"],
        "Env": {
          "NODE_ENV": "production",
          "LOG_LEVEL": "info"
        },
        "WorkingDirectory": "/opt/mcp-tools/code-analyzer"
      }
    },
    "Security": {
      "RequireAuthentication": true,
      "TokenValidation": {
        "Issuer": "hypercube-api",
        "Audience": "hypercube-clients",
        "ClockSkew": "00:05:00"
      },
      "GlobalRateLimit": {
        "RequestsPerMinute": 1000,
        "RequestsPerHour": 10000
      },
      "ContentFiltering": {
        "Enabled": true,
        "Filters": ["profanity", "personal-info", "malicious-content"]
      }
    },
    "Resilience": {
      "MaxRetries": 3,
      "CircuitBreaker": {
        "FailureThreshold": 5,
        "RecoveryTimeout": "00:01:00"
      },
      "Timeout": "00:00:30"
    }
  },
  "ConnectionStrings": {
    "HypercubeDatabase": "Host=localhost;Port=5432;Database=hypercube;Username=hypercube_user;Password=${DB_PASSWORD};SSL Mode=Require;Trust Server Certificate=false"
  },
  "HealthChecks": {
    "UI": {
      "Enabled": true,
      "Port": 5002
    }
  },
  "OpenTelemetry": {
    "ServiceName": "hypercube-mcp-api",
    "ServiceVersion": "1.0.0",
    "Traces": {
      "Exporter": "otlp",
      "Endpoint": "http://otel-collector:4317"
    },
    "Metrics": {
      "Exporter": "otlp",
      "Endpoint": "http://otel-collector:4317",
      "Interval": "00:00:15"
    }
  }
}
```

## Database Setup

### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE hypercube;
CREATE USER hypercube_user WITH ENCRYPTED PASSWORD '${DB_PASSWORD}';
GRANT ALL PRIVILEGES ON DATABASE hypercube TO hypercube_user;

-- Enable required extensions
\c hypercube;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS hypercube;
CREATE EXTENSION IF NOT EXISTS semantic_ops;
CREATE EXTENSION IF NOT EXISTS embedding_ops;
CREATE EXTENSION IF NOT EXISTS generative;

-- Grant schema permissions
GRANT USAGE ON SCHEMA public TO hypercube_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hypercube_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hypercube_user;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_atom_geom ON atom USING GIST (geom);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_composition_centroid ON composition USING GIST (centroid);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_relation_endpoints ON relation (source_id, target_id);
```

### Database Migration
```bash
# Run database migrations
dotnet ef database update --project HypercubeGenerativeApi

# Seed initial data (atoms, etc.)
dotnet run -- seed-atoms

# Verify database connectivity
dotnet run -- check-db
```

## Application Deployment

### Source Code Deployment
```bash
# Clone repository
git clone https://github.com/AHartTN/Hartonomous-Opus.git
cd Hartonomous-Opus/csharp/HypercubeGenerativeApi

# Checkout specific version
git checkout v1.0.0-mcp

# Build application
dotnet restore
dotnet build --configuration Release --no-restore
dotnet publish --configuration Release --output /opt/hypercube-api --no-build
```

### Service Configuration
```ini
# /etc/systemd/system/hypercube-api.service
[Unit]
Description=Hartonomous-Opus Hypercube API with MCP Support
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=hypercube
Group=hypercube
WorkingDirectory=/opt/hypercube-api
ExecStart=/usr/bin/dotnet HypercubeGenerativeApi.dll
Restart=always
RestartSec=10
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=DOTNET_CLI_TELEMETRY_OPTOUT=1
EnvironmentFile=/etc/hypercube/hypercube.env

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/hypercube-api /var/log/hypercube
ProtectHome=true
PrivateDevices=true

# Resource limits
MemoryLimit=2G
CPUQuota=200%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hypercube-api

[Install]
WantedBy=multi-user.target
```

### SSL/TLS Configuration
```bash
# Generate self-signed certificate (for testing only)
openssl req -x509 -newkey rsa:4096 -keyout /etc/ssl/private/hypercube.key -out /etc/ssl/certs/hypercube.crt -days 365 -nodes -subj "/CN=hypercube.local"

# For production, obtain certificate from Let's Encrypt
certbot certonly --standalone -d hypercube.yourdomain.com

# Update nginx configuration
server {
    listen 443 ssl http2;
    server_name hypercube.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/hypercube.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hypercube.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /mcp {
        proxy_pass http://localhost:5001/mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name hypercube.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

## Containerized Deployment

### Dockerfile
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0-jammy AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443
EXPOSE 5001

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM mcr.microsoft.com/dotnet/sdk:8.0-jammy AS build
WORKDIR /src

# Copy csproj files and restore
COPY ["HypercubeGenerativeApi.csproj", "."]
RUN dotnet restore "HypercubeGenerativeApi.csproj"

# Copy everything else and build
COPY . .
RUN dotnet build "HypercubeGenerativeApi.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "HypercubeGenerativeApi.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .

# Create non-root user
RUN groupadd -r hypercube && useradd -r -g hypercube hypercube
RUN chown -R hypercube:hypercube /app
USER hypercube

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

ENTRYPOINT ["dotnet", "HypercubeGenerativeApi.dll"]
```

### Docker Compose Configuration
```yaml
version: '3.8'

services:
  hypercube-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
      - "5001:5001"
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      - ASPNETCORE_URLS=https://+:443;http://+:80
      - ConnectionStrings__HypercubeDatabase=Host=db;Database=hypercube;Username=hypercube_user;Password=${DB_PASSWORD}
      - MCP__Server__Enabled=true
      - MCP__Server__Transport__Port=5001
    env_file:
      - .env
    depends_on:
      - db
    volumes:
      - ./logs:/app/logs
      - ./certs:/app/certs:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=hypercube
      - POSTGRES_USER=hypercube_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hypercube_user -d hypercube"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - hypercube-api
    restart: unless-stopped

  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml:ro
    ports:
      - "4317:4317"
    depends_on:
      - hypercube-api
    restart: unless-stopped

volumes:
  postgres_data:

networks:
  default:
    name: hypercube-network
```

## Kubernetes Deployment

### Namespace and RBAC
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hypercube
  labels:
    name: hypercube

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: hypercube-api
  namespace: hypercube

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hypercube-api
  namespace: hypercube
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hypercube-api
  namespace: hypercube
subjects:
- kind: ServiceAccount
  name: hypercube-api
  namespace: hypercube
roleRef:
  kind: Role
  name: hypercube-api
  apiGroup: rbac.authorization.k8s.io
```

### Database Deployment
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hypercube-db
  namespace: hypercube
spec:
  serviceName: hypercube-db
  replicas: 1
  selector:
    matchLabels:
      app: hypercube-db
  template:
    metadata:
      labels:
        app: hypercube-db
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: "hypercube"
        - name: POSTGRES_USER
          value: "hypercube_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: hypercube-secrets
              key: db-password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        - name: init-scripts
          mountPath: /docker-entrypoint-initdb.d
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - hypercube_user
            - -d
            - hypercube
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - hypercube_user
            - -d
            - hypercube
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
    name: postgres-data
    namespace: hypercube
  spec:
    accessModes: ["ReadWriteOnce"]
    resources:
      requests:
        storage: 50Gi

---
apiVersion: v1
kind: Service
metadata:
  name: hypercube-db
  namespace: hypercube
spec:
  selector:
    app: hypercube-db
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

### API Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypercube-api
  namespace: hypercube
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypercube-api
  template:
    metadata:
      labels:
        app: hypercube-api
    spec:
      serviceAccountName: hypercube-api
      containers:
      - name: api
        image: hypercube/hypercube-api:1.0.0-mcp
        ports:
        - containerPort: 80
          name: http
        - containerPort: 443
          name: https
        - containerPort: 5001
          name: mcp
        env:
        - name: ASPNETCORE_ENVIRONMENT
          value: "Production"
        - name: ASPNETCORE_URLS
          value: "http://+:80;https://+:443"
        - name: ConnectionStrings__HypercubeDatabase
          valueFrom:
            secretKeyRef:
              name: hypercube-secrets
              key: database-connection
        - name: MCP__Server__Enabled
          value: "true"
        - name: MCP__Server__Transport__Port
          value: "5001"
        - name: JWT__Key
          valueFrom:
            secretKeyRef:
              name: hypercube-secrets
              key: jwt-key
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: certs
          mountPath: /app/certs
          readOnly: true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
            scheme: HTTPS
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 80
            scheme: HTTPS
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
      volumes:
      - name: logs
        emptyDir: {}
      - name: certs
        secret:
          secretName: hypercube-tls-cert

---
apiVersion: v1
kind: Service
metadata:
  name: hypercube-api
  namespace: hypercube
spec:
  selector:
    app: hypercube-api
  ports:
  - name: http
    port: 80
    targetPort: 80
  - name: https
    port: 443
    targetPort: 443
  - name: mcp
    port: 5001
    targetPort: 5001
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hypercube-api
  namespace: hypercube
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.hypercube.yourdomain.com
    - mcp.hypercube.yourdomain.com
    secretName: hypercube-tls
  rules:
  - host: api.hypercube.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hypercube-api
            port:
              number: 80
  - host: mcp.hypercube.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hypercube-api
            port:
              number: 5001
```

### Secrets Management
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hypercube-secrets
  namespace: hypercube
type: Opaque
data:
  # Base64 encoded values
  db-password: <base64-encoded-password>
  jwt-key: <base64-encoded-jwt-key>
  database-connection: <base64-encoded-connection-string>
  external-tools-token: <base64-encoded-token>
```

## Monitoring and Observability

### OpenTelemetry Configuration
```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  jaeger:
    endpoint: "jaeger:14268"
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hypercube-api'
    static_configs:
      - targets: ['hypercube-api:80']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
```

### Grafana Dashboards
Key metrics to monitor:
- MCP request rate and latency
- Tool usage statistics
- Error rates by tool and client
- Authentication failures
- Rate limiting events
- Database connection pool usage
- Memory and CPU usage

## Backup and Recovery

### Database Backup
```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/var/backups/hypercube"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h localhost -U hypercube_user -d hypercube -Fc > "${BACKUP_DIR}/hypercube_${DATE}.dump"

# Compress
gzip "${BACKUP_DIR}/hypercube_${DATE}.dump"

# Clean old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "hypercube_*.dump.gz" -mtime +7 -delete

# Upload to remote storage (optional)
# aws s3 cp "${BACKUP_DIR}/hypercube_${DATE}.dump.gz" s3://hypercube-backups/
```

### Application Backup
```bash
# Configuration backup
tar -czf /var/backups/hypercube-config-$(date +%Y%m%d).tar.gz \
    /etc/hypercube/ \
    /opt/hypercube-api/appsettings.Production.json

# Log backup (if not using centralized logging)
tar -czf /var/backups/hypercube-logs-$(date +%Y%m%d).tar.gz \
    /var/log/hypercube/
```

### Recovery Procedures
```bash
# Database recovery
pg_restore -h localhost -U hypercube_user -d hypercube -c /var/backups/hypercube_20231201.dump

# Application recovery
systemctl stop hypercube-api
tar -xzf /var/backups/hypercube-config-20231201.tar.gz -C /
systemctl start hypercube-api

# Verify recovery
curl https://localhost/health
curl https://localhost/mcp -H "Authorization: Bearer ${TEST_TOKEN}"
```

## Scaling Considerations

### Horizontal Scaling
- **Database**: Read replicas for query-heavy operations
- **API**: Multiple instances behind load balancer
- **MCP**: Stateless design supports horizontal scaling
- **Session Management**: External session store (Redis) for consistency

### Vertical Scaling
- **Memory**: Increase for larger semantic operations
- **CPU**: Additional cores for parallel processing
- **Storage**: SSD storage for better I/O performance

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hypercube-api-hpa
  namespace: hypercube
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hypercube-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

#### MCP Server Not Starting
```bash
# Check logs
journalctl -u hypercube-api -f

# Verify configuration
dotnet run -- check-config

# Test MCP endpoint
curl -H "Authorization: Bearer ${TOKEN}" https://localhost/mcp
```

#### Database Connection Issues
```bash
# Test database connectivity
psql -h localhost -U hypercube_user -d hypercube -c "SELECT 1;"

# Check connection pool
curl https://localhost/health | jq '.data.database'
```

#### High Latency
```bash
# Check system resources
top -p $(pgrep dotnet)

# Monitor database performance
psql -c "SELECT * FROM pg_stat_activity;"

# Enable detailed logging
export Logging__LogLevel__HypercubeGenerativeApi=Debug
systemctl restart hypercube-api
```

#### Authentication Failures
```bash
# Validate JWT token
curl -H "Authorization: Bearer ${TOKEN}" https://localhost/debug/token

# Check token expiration
curl https://localhost/debug/jwt-expiry

# Verify user permissions
curl -H "Authorization: Bearer ${TOKEN}" https://localhost/debug/permissions
```

### Performance Tuning

#### Database Optimization
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM semantic_search('query', 10, 0.1);

-- Update statistics
ANALYZE VERBOSE;

-- Check index usage
SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;
```

#### Application Tuning
```bash
# Profile application
dotnet-trace collect --process-id $(pgrep dotnet) --output trace.nettrace

# Memory analysis
dotnet-dump collect --process-id $(pgrep dotnet) --output dump.dmp
dotnet-dump analyze dump.dmp
```

### Monitoring Commands
```bash
# Health check
curl https://localhost/health

# MCP server status
curl https://localhost/mcp/status

# Metrics
curl https://localhost/metrics

# Tool usage statistics
curl -H "Authorization: Bearer ${ADMIN_TOKEN}" https://localhost/admin/mcp/stats
```

## Security Checklist

### Pre-Deployment
- [ ] SSL/TLS certificates installed and configured
- [ ] Authentication system tested
- [ ] Authorization policies defined
- [ ] Rate limiting configured
- [ ] Input validation active
- [ ] Content filtering enabled
- [ ] Audit logging enabled

### Post-Deployment
- [ ] Security headers verified
- [ ] Vulnerability scanning completed
- [ ] Penetration testing performed
- [ ] Incident response plan documented
- [ ] Backup procedures tested
- [ ] Monitoring alerts configured

### Ongoing Maintenance
- [ ] Regular security updates applied
- [ ] Log monitoring and analysis
- [ ] Performance monitoring active
- [ ] Backup verification
- [ ] Access review and rotation

This deployment guide provides comprehensive instructions for successfully deploying MCP functionality in production environments, with considerations for security, scalability, monitoring, and maintenance.