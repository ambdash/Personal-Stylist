from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter(
    "bot_requests_total",
    "Total number of requests",
    ["command", "status"]
)

REQUEST_LATENCY = Histogram(
    "bot_request_latency_seconds",
    "Request latency in seconds",
    ["command"]
)

# Model metrics
MODEL_INFERENCE_COUNT = Counter(
    "model_inference_total",
    "Total number of model inferences",
    ["model_name", "status"]
)

MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_name"]
)

# Neo4j metrics
NEO4J_QUERY_COUNT = Counter(
    "neo4j_queries_total",
    "Total number of Neo4j queries",
    ["query_type", "status"]
)

NEO4J_QUERY_LATENCY = Histogram(
    "neo4j_query_latency_seconds",
    "Neo4j query latency in seconds",
    ["query_type"]
)

# System metrics
ACTIVE_USERS = Gauge(
    "bot_active_users",
    "Number of active users in the last 5 minutes"
)

QUEUE_SIZE = Gauge(
    "celery_queue_size",
    "Number of tasks in Celery queue",
    ["queue_name"]
)

def track_request_metric(command: str, status: str = "success"):
    """Track bot request metrics"""
    REQUEST_COUNT.labels(command=command, status=status).inc()

def track_model_metric(model_name: str, latency: float, status: str = "success"):
    """Track model inference metrics"""
    MODEL_INFERENCE_COUNT.labels(model_name=model_name, status=status).inc()
    MODEL_INFERENCE_LATENCY.labels(model_name=model_name).observe(latency)

def track_neo4j_metric(query_type: str, latency: float, status: str = "success"):
    """Track Neo4j query metrics"""
    NEO4J_QUERY_COUNT.labels(query_type=query_type, status=status).inc()
    NEO4J_QUERY_LATENCY.labels(query_type=query_type).observe(latency)

class MetricsTimer:
    """Context manager for timing operations"""
    def __init__(self, metric_type: str, labels: dict):
        self.start_time = None
        self.metric_type = metric_type
        self.labels = labels

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.metric_type == "model":
            track_model_metric(self.labels["model_name"], duration)
        elif self.metric_type == "neo4j":
            track_neo4j_metric(self.labels["query_type"], duration)
        elif self.metric_type == "request":
            REQUEST_LATENCY.labels(command=self.labels["command"]).observe(duration) 