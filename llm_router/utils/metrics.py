# llm_router/utils/metrics.py

"""Performance metrics for llm-router."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import structlog

from ..types import RouterStats

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize the metrics collector."""
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.start_time = datetime.utcnow()
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        self.metrics[f"counter_{name}"].append({
            "timestamp": datetime.utcnow(),
            "value": self.counters[name]
        })
    
    def record_timer(self, name: str, duration_ms: float) -> None:
        """Record a timer metric."""
        self.timers[name].append(duration_ms)
        if len(self.timers[name]) > self.max_history:
            self.timers[name] = self.timers[name][-self.max_history:]
        
        self.metrics[f"timer_{name}"].append({
            "timestamp": datetime.utcnow(),
            "value": duration_ms
        })
    
    def record_gauge(self, name: str, value: float) -> None:
        """Record a gauge metric."""
        self.metrics[f"gauge_{name}"].append({
            "timestamp": datetime.utcnow(),
            "value": value
        })
    
    def record_request(
        self,
        provider: str,
        model: str,
        success: bool,
        duration_ms: float,
        tokens_used: int = 0,
        cost: float = 0.0
    ) -> None:
        """Record a complete request metric."""
        # Increment counters
        self.increment_counter("total_requests")
        self.increment_counter(f"requests_{provider}")
        self.increment_counter(f"requests_{model}")
        
        if success:
            self.increment_counter("successful_requests")
            self.increment_counter(f"successful_requests_{provider}")
        else:
            self.increment_counter("failed_requests")
            self.increment_counter(f"failed_requests_{provider}")
        
        # Record timers
        self.record_timer("request_duration", duration_ms)
        self.record_timer(f"request_duration_{provider}", duration_ms)
        self.record_timer(f"request_duration_{model}", duration_ms)
        
        # Record gauges
        if tokens_used > 0:
            self.record_gauge("tokens_used", tokens_used)
            self.record_gauge(f"tokens_used_{provider}", tokens_used)
        
        if cost > 0:
            self.record_gauge("cost", cost)
            self.record_gauge(f"cost_{provider}", cost)
    
    def record_cache_hit(self, provider: str) -> None:
        """Record a cache hit."""
        self.increment_counter("cache_hits")
        self.increment_counter(f"cache_hits_{provider}")
    
    def record_cache_miss(self, provider: str) -> None:
        """Record a cache miss."""
        self.increment_counter("cache_misses")
        self.increment_counter(f"cache_misses_{provider}")
    
    def get_counter(self, name: str) -> int:
        """Get current value of a counter."""
        return self.counters.get(name, 0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timer."""
        values = self.timers.get(name, [])
        if not values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_values = sorted(values)
        count = len(values)
        
        return {
            "count": count,
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / count,
            "median": sorted_values[count // 2],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_gauge_latest(self, name: str) -> Optional[float]:
        """Get the latest value of a gauge."""
        gauge_data = self.metrics.get(f"gauge_{name}")
        if gauge_data:
            return gauge_data[-1]["value"]
        return None
    
    def get_gauge_history(self, name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get gauge history for the specified time period."""
        gauge_data = self.metrics.get(f"gauge_{name}")
        if not gauge_data:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            entry for entry in gauge_data
            if entry["timestamp"] >= cutoff_time
        ]
    
    def get_router_stats(self) -> RouterStats:
        """Get comprehensive router statistics."""
        total_requests = self.get_counter("total_requests")
        successful_requests = self.get_counter("successful_requests")
        failed_requests = self.get_counter("failed_requests")
        cache_hits = self.get_counter("cache_hits")
        cache_misses = self.get_counter("cache_misses")
        
        # Calculate average latency
        request_duration_stats = self.get_timer_stats("request_duration")
        average_latency = request_duration_stats.get("mean", 0.0)
        
        # Get provider-specific stats
        provider_stats = {}
        for metric_name in self.metrics.keys():
            if metric_name.startswith("requests_") and metric_name != "requests_total":
                provider = metric_name.replace("requests_", "")
                provider_stats[provider] = {
                    "total_requests": self.get_counter(f"requests_{provider}"),
                    "successful_requests": self.get_counter(f"successful_requests_{provider}"),
                    "failed_requests": self.get_counter(f"failed_requests_{provider}"),
                    "cache_hits": self.get_counter(f"cache_hits_{provider}"),
                    "cache_misses": self.get_counter(f"cache_misses_{provider}"),
                    "average_latency": self.get_timer_stats(f"request_duration_{provider}").get("mean", 0.0)
                }
        
        return RouterStats(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            average_latency_ms=average_latency,
            provider_stats=provider_stats,
            created_at=datetime.utcnow()
        )
    
    def get_provider_performance(self, provider: str) -> Dict[str, Any]:
        """Get performance metrics for a specific provider."""
        return {
            "total_requests": self.get_counter(f"requests_{provider}"),
            "successful_requests": self.get_counter(f"successful_requests_{provider}"),
            "failed_requests": self.get_counter(f"failed_requests_{provider}"),
            "success_rate": self._calculate_success_rate(provider),
            "cache_hits": self.get_counter(f"cache_hits_{provider}"),
            "cache_misses": self.get_counter(f"cache_misses_{provider}"),
            "cache_hit_rate": self._calculate_cache_hit_rate(provider),
            "latency_stats": self.get_timer_stats(f"request_duration_{provider}"),
            "total_cost": self.get_gauge_latest(f"cost_{provider}") or 0.0,
            "total_tokens": self.get_gauge_latest(f"tokens_used_{provider}") or 0
        }
    
    def _calculate_success_rate(self, provider: str) -> float:
        """Calculate success rate for a provider."""
        total = self.get_counter(f"requests_{provider}")
        successful = self.get_counter(f"successful_requests_{provider}")
        
        if total == 0:
            return 0.0
        
        return successful / total
    
    def _calculate_cache_hit_rate(self, provider: str) -> float:
        """Calculate cache hit rate for a provider."""
        hits = self.get_counter(f"cache_hits_{provider}")
        misses = self.get_counter(f"cache_misses_{provider}")
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in the specified format."""
        if format.lower() == "json":
            import json
            
            export_data = {
                "counters": dict(self.counters),
                "timers": {name: self.get_timer_stats(name) for name in self.timers.keys()},
                "gauges": {name: self.get_gauge_latest(name) for name in self.metrics.keys() if name.startswith("gauge_")},
                "start_time": self.start_time.isoformat(),
                "export_time": datetime.utcnow().isoformat()
            }
            
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "prometheus":
            # Export in Prometheus format
            lines = []
            
            # Counters
            for name, value in self.counters.items():
                lines.append(f'# TYPE {name} counter')
                lines.append(f'{name} {value}')
            
            # Gauges
            for name in self.metrics.keys():
                if name.startswith("gauge_"):
                    gauge_name = name.replace("gauge_", "")
                    value = self.get_gauge_latest(name) or 0
                    lines.append(f'# TYPE {gauge_name} gauge')
                    lines.append(f'{gauge_name} {value}')
            
            # Histograms (from timers)
            for name in self.timers.keys():
                stats = self.get_timer_stats(name)
                lines.append(f'# TYPE {name}_duration_seconds histogram')
                lines.append(f'{name}_duration_seconds_count {stats["count"]}')
                lines.append(f'{name}_duration_seconds_sum {sum(self.timers[name]) / 1000}')  # Convert to seconds
                lines.append(f'{name}_duration_seconds_bucket{{le="0.1"}} {len([v for v in self.timers[name] if v <= 100])}')
                lines.append(f'{name}_duration_seconds_bucket{{le="0.5"}} {len([v for v in self.timers[name] if v <= 500])}')
                lines.append(f'{name}_duration_seconds_bucket{{le="1.0"}} {len([v for v in self.timers[name] if v <= 1000])}')
                lines.append(f'{name}_duration_seconds_bucket{{le="+Inf"}} {len(self.timers[name])}')
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.start_time = datetime.utcnow()
        logger.info("Metrics reset") 