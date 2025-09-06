"""Monitoring and metrics collection"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a single metric"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    

@dataclass
class HealthStatus:
    """Health check status"""
    healthy: bool
    checks: Dict[str, bool]
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collect and store application metrics"""
    
    def __init__(self, max_metrics: int = 10000, retention_minutes: int = 60):
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics))
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.retention_minutes = retention_minutes
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
    
    def start(self):
        """Start metrics collection"""
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop(self):
        """Stop metrics collection"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     unit: str = ""):
        """Record a metric"""
        metric = Metric(name=name, value=value, tags=tags or {}, unit=unit)
        
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.metrics[key].append(metric)
    
    def increment_counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.histograms[key].append(value)
            # Keep only last 1000 values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def get_metrics(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None,
                    last_minutes: int = 5) -> List[Metric]:
        """Get metrics for the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=last_minutes)
        result = []
        
        with self._lock:
            for key, metrics_deque in self.metrics.items():
                if name and not key.startswith(name):
                    continue
                
                for metric in metrics_deque:
                    if metric.timestamp >= cutoff_time:
                        if tags and not all(metric.tags.get(k) == v for k, v in tags.items()):
                            continue
                        result.append(metric)
        
        return result
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            return self.counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            return self.gauges.get(key, 0)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            values = self.histograms.get(key, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'p50': sorted_values[len(sorted_values) // 2],
                'p95': sorted_values[int(len(sorted_values) * 0.95)],
                'p99': sorted_values[int(len(sorted_values) * 0.99)]
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self.counters.items():
                name, tags = self._parse_metric_key(key)
                tags_str = self._format_prometheus_tags(tags)
                lines.append(f"{name}_total{tags_str} {value}")
            
            # Export gauges
            for key, value in self.gauges.items():
                name, tags = self._parse_metric_key(key)
                tags_str = self._format_prometheus_tags(tags)
                lines.append(f"{name}{tags_str} {value}")
            
            # Export histogram stats
            for key, values in self.histograms.items():
                if values:
                    name, tags = self._parse_metric_key(key)
                    stats = self.get_histogram_stats(name, tags)
                    tags_str = self._format_prometheus_tags(tags)
                    
                    for stat_name, stat_value in stats.items():
                        lines.append(f"{name}_{stat_name}{tags_str} {stat_value}")
        
        return '\n'.join(lines)
    
    def _get_metric_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Generate metric key from name and tags"""
        if not tags:
            return name
        tags_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tags_str}"
    
    def _parse_metric_key(self, key: str) -> tuple:
        """Parse metric key into name and tags"""
        parts = key.split(',', 1)
        name = parts[0]
        tags = {}
        
        if len(parts) > 1:
            for tag in parts[1].split(','):
                k, v = tag.split('=', 1)
                tags[k] = v
        
        return name, tags
    
    def _format_prometheus_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for Prometheus"""
        if not tags:
            return ""
        tags_str = ','.join(f'{k}="{v}"' for k, v in sorted(tags.items()))
        return f"{{{tags_str}}}"
    
    def _cleanup_loop(self):
        """Background thread to clean up old metrics"""
        while not self._stop_cleanup.wait(60):  # Check every minute
            self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.retention_minutes)
        
        with self._lock:
            for key in list(self.metrics.keys()):
                # Remove old metrics from deque
                while self.metrics[key] and self.metrics[key][0].timestamp < cutoff_time:
                    self.metrics[key].popleft()
                
                # Remove empty deques
                if not self.metrics[key]:
                    del self.metrics[key]


class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def start(self, interval: int = 60):
        """Start system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop system monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv
            }
        }
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while not self._stop_event.wait(interval):
            try:
                stats = self.get_system_stats()
                
                # Record metrics
                self.metrics_collector.set_gauge('system.cpu.percent', stats['cpu']['percent'])
                self.metrics_collector.set_gauge('system.memory.percent', stats['memory']['percent'])
                self.metrics_collector.set_gauge('system.disk.percent', stats['disk']['percent'])
                self.metrics_collector.set_gauge('system.network.bytes_sent', stats['network']['bytes_sent'])
                self.metrics_collector.set_gauge('system.network.bytes_recv', stats['network']['bytes_recv'])
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")


class HealthChecker:
    """Application health checks"""
    
    def __init__(self):
        self.checks = {}
        self._last_check_results = {}
        self._check_timeout = 5  # seconds
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check"""
        self.checks[name] = check_func
    
    def unregister_check(self, name: str):
        """Unregister a health check"""
        if name in self.checks:
            del self.checks[name]
    
    def check_health(self) -> HealthStatus:
        """Run all health checks"""
        results = {}
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                # Run check with timeout
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(check_func)
                    result = future.result(timeout=self._check_timeout)
                    results[name] = bool(result)
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
            
            if not results[name]:
                all_healthy = False
        
        self._last_check_results = results
        
        return HealthStatus(
            healthy=all_healthy,
            checks=results,
            message="All checks passed" if all_healthy else "Some checks failed",
            details={
                'total_checks': len(self.checks),
                'passed': sum(1 for v in results.values() if v),
                'failed': sum(1 for v in results.values() if not v)
            }
        )
    
    def get_last_results(self) -> Dict[str, bool]:
        """Get last health check results"""
        return self._last_check_results.copy()


class PerformanceTracker:
    """Track operation performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._active_operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_type: str, **tags):
        """Start tracking an operation"""
        with self._lock:
            self._active_operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'tags': tags
            }
    
    def end_operation(self, operation_id: str, success: bool = True, **additional_tags):
        """End tracking an operation"""
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Operation {operation_id} not found")
                return
            
            operation = self._active_operations.pop(operation_id)
            duration = time.time() - operation['start_time']
            
            # Merge tags
            tags = {**operation['tags'], **additional_tags, 'success': str(success)}
            
            # Record metrics
            self.metrics_collector.record_histogram(
                f"operation.{operation['type']}.duration",
                duration,
                tags
            )
            self.metrics_collector.increment_counter(
                f"operation.{operation['type']}.count",
                tags=tags
            )
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of active operations"""
        with self._lock:
            return [
                {
                    'id': op_id,
                    'type': op_data['type'],
                    'duration': time.time() - op_data['start_time'],
                    'tags': op_data['tags']
                }
                for op_id, op_data in self._active_operations.items()
            ]