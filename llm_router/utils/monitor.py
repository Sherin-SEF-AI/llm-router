# llm_router/utils/monitor.py

"""Health monitoring for llm-router."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import structlog

from ..types import HealthStatus
from ..exceptions import ProviderError

logger = structlog.get_logger(__name__)


class HealthMonitor:
    """Monitors health and availability of LLM providers."""
    
    def __init__(self, check_interval: int = 300):
        """Initialize the health monitor."""
        self.check_interval = check_interval
        self.health_status: Dict[str, HealthStatus] = {}
        self.provider_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start_monitoring(self) -> None:
        """Start the health monitoring background task."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started", interval=self.check_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop the health monitoring background task."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._check_all_providers()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_all_providers(self) -> None:
        """Check health of all registered providers."""
        for provider_name in self.health_status.keys():
            await self._check_provider_health(provider_name)
    
    async def _check_provider_health(self, provider_name: str) -> None:
        """Check health of a specific provider."""
        try:
            # This would be implemented by the actual provider
            # For now, we'll simulate a health check
            start_time = time.time()
            
            # Simulate health check (replace with actual provider health check)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Update health status
            self._update_health_status(provider_name, True, response_time)
            
        except Exception as e:
            logger.warning(
                "Provider health check failed",
                provider=provider_name,
                error=str(e)
            )
            self._update_health_status(provider_name, False, None, str(e))
    
    def _update_health_status(
        self,
        provider: str,
        healthy: bool,
        response_time: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """Update health status for a provider."""
        current_status = self.health_status.get(provider)
        
        if current_status is None:
            current_status = HealthStatus(
                provider=provider,
                healthy=healthy,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                error_count=0 if healthy else 1,
                success_rate=1.0 if healthy else 0.0
            )
        else:
            # Update existing status
            error_count = current_status.error_count
            if not healthy:
                error_count += 1
            else:
                error_count = max(0, error_count - 1)  # Gradually recover
            
            # Calculate success rate based on recent checks
            success_rate = max(0.0, 1.0 - (error_count / 10.0))
            
            current_status = HealthStatus(
                provider=provider,
                healthy=healthy,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                error_count=error_count,
                success_rate=success_rate
            )
        
        self.health_status[provider] = current_status
        
        logger.info(
            "Health status updated",
            provider=provider,
            healthy=healthy,
            response_time=response_time,
            error_count=current_status.error_count,
            success_rate=current_status.success_rate
        )
    
    def register_provider(self, provider_name: str) -> None:
        """Register a provider for health monitoring."""
        if provider_name not in self.health_status:
            self.health_status[provider_name] = HealthStatus(
                provider=provider_name,
                healthy=True,
                last_check=datetime.utcnow(),
                response_time_ms=None,
                error_count=0,
                success_rate=1.0
            )
            logger.info("Provider registered for health monitoring", provider=provider_name)
    
    def unregister_provider(self, provider_name: str) -> None:
        """Unregister a provider from health monitoring."""
        if provider_name in self.health_status:
            del self.health_status[provider_name]
            logger.info("Provider unregistered from health monitoring", provider=provider_name)
    
    def get_health_status(self, provider: str) -> Optional[HealthStatus]:
        """Get health status for a specific provider."""
        return self.health_status.get(provider)
    
    def get_all_health_statuses(self) -> Dict[str, HealthStatus]:
        """Get health status for all providers."""
        return self.health_status.copy()
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of currently healthy providers."""
        return [
            provider for provider, status in self.health_status.items()
            if status.healthy
        ]
    
    def get_unhealthy_providers(self) -> List[str]:
        """Get list of currently unhealthy providers."""
        return [
            provider for provider, status in self.health_status.items()
            if not status.healthy
        ]
    
    def record_request_result(
        self,
        provider: str,
        success: bool,
        response_time_ms: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """Record the result of a request to update health metrics."""
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": [],
                "last_request": None
            }
        
        stats = self.provider_stats[provider]
        stats["total_requests"] += 1
        stats["last_request"] = datetime.utcnow()
        
        if success:
            stats["successful_requests"] += 1
            if response_time_ms is not None:
                stats["response_times"].append(response_time_ms)
                # Keep only last 100 response times
                if len(stats["response_times"]) > 100:
                    stats["response_times"] = stats["response_times"][-100:]
        else:
            stats["failed_requests"] += 1
        
        # Update health status based on request result
        self._update_health_status(provider, success, response_time_ms, error)
    
    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """Get detailed statistics for a provider."""
        if provider not in self.provider_stats:
            return {}
        
        stats = self.provider_stats[provider]
        response_times = stats.get("response_times", [])
        
        return {
            "total_requests": stats["total_requests"],
            "successful_requests": stats["successful_requests"],
            "failed_requests": stats["failed_requests"],
            "success_rate": stats["successful_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "last_request": stats["last_request"],
            "health_status": self.health_status.get(provider)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        return {
            provider: self.get_provider_stats(provider)
            for provider in self.provider_stats.keys()
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.provider_stats.clear()
        logger.info("Health monitoring statistics reset") 