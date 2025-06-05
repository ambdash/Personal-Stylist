from prometheus_client import Counter

# Define metrics
request_counter = Counter(
    'bot_requests_total',
    'Total number of bot requests',
    ['request_type', 'style']
)

def track_request(request_type: str, style: str = None):
    """Track a request in Prometheus metrics"""
    request_counter.labels(request_type=request_type, style=style or 'unknown').inc() 