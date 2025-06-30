# llm_router/utils/cost_tracker.py

"""Cost tracking for llm-router."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import structlog

from ..types import CostSummary, ModelConfig
from ..exceptions import LLMRouterError

logger = structlog.get_logger(__name__)


class CostTracker:
    """Tracks costs across different providers and models."""
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.costs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.model_configs: Dict[str, ModelConfig] = {}
        self.start_time = datetime.utcnow()
    
    def add_model_config(self, model_config: ModelConfig) -> None:
        """Add a model configuration for cost calculation."""
        self.model_configs[model_config.name] = model_config
    
    def record_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None
    ) -> None:
        """Record a cost entry."""
        timestamp = datetime.utcnow()
        
        # Calculate cost if not provided
        if cost is None:
            cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        entry = {
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost
        }
        
        self.costs[provider].append(entry)
        
        logger.info(
            "Cost recorded",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model configuration."""
        if model not in self.model_configs:
            logger.warning(f"No cost configuration for model: {model}")
            return 0.0
        
        config = self.model_configs[model]
        
        input_cost = (input_tokens / 1000) * config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * config.output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_cost_summary(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> CostSummary:
        """Get cost summary for a time period."""
        if period_start is None:
            period_start = self.start_time
        if period_end is None:
            period_end = datetime.utcnow()
        
        provider_costs = defaultdict(float)
        model_costs = defaultdict(float)
        total_cost = 0.0
        request_count = 0
        
        # Aggregate costs across all providers
        for provider, entries in self.costs.items():
            for entry in entries:
                if period_start <= entry["timestamp"] <= period_end:
                    cost = entry["cost"]
                    provider_costs[provider] += cost
                    model_costs[entry["model"]] += cost
                    total_cost += cost
                    request_count += 1
        
        return CostSummary(
            total_cost=total_cost,
            provider_costs=dict(provider_costs),
            model_costs=dict(model_costs),
            period_start=period_start,
            period_end=period_end,
            request_count=request_count
        )
    
    def get_provider_costs(self, provider: str) -> List[Dict[str, Any]]:
        """Get all cost entries for a specific provider."""
        return self.costs.get(provider, [])
    
    def get_model_costs(self, model: str) -> List[Dict[str, Any]]:
        """Get all cost entries for a specific model."""
        model_entries = []
        for provider_entries in self.costs.values():
            for entry in provider_entries:
                if entry["model"] == model:
                    model_entries.append(entry)
        return model_entries
    
    def get_daily_costs(self, days: int = 30) -> List[CostSummary]:
        """Get daily cost summaries for the last N days."""
        summaries = []
        end_date = datetime.utcnow()
        
        for i in range(days):
            day_start = end_date - timedelta(days=i+1)
            day_end = end_date - timedelta(days=i)
            summary = self.get_cost_summary(day_start, day_end)
            summaries.append(summary)
        
        return summaries
    
    def get_monthly_costs(self, months: int = 12) -> List[CostSummary]:
        """Get monthly cost summaries for the last N months."""
        summaries = []
        end_date = datetime.utcnow()
        
        for i in range(months):
            month_start = end_date - timedelta(days=30*(i+1))
            month_end = end_date - timedelta(days=30*i)
            summary = self.get_cost_summary(month_start, month_end)
            summaries.append(summary)
        
        return summaries
    
    def reset_costs(self) -> None:
        """Reset all cost tracking data."""
        self.costs.clear()
        self.start_time = datetime.utcnow()
        logger.info("Cost tracking data reset")
    
    def export_costs(self, format: str = "json") -> str:
        """Export cost data in the specified format."""
        if format.lower() == "json":
            import json
            return json.dumps(self.costs, default=str, indent=2)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "timestamp", "provider", "model", "input_tokens",
                "output_tokens", "total_tokens", "cost"
            ])
            
            # Write data
            for provider_entries in self.costs.values():
                for entry in provider_entries:
                    writer.writerow([
                        entry["timestamp"],
                        entry["provider"],
                        entry["model"],
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["total_tokens"],
                        entry["cost"]
                    ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost tracking statistics."""
        total_entries = sum(len(entries) for entries in self.costs.values())
        total_cost = sum(
            sum(entry["cost"] for entry in entries)
            for entries in self.costs.values()
        )
        
        return {
            "total_entries": total_entries,
            "total_cost": total_cost,
            "providers": list(self.costs.keys()),
            "models": list(self.model_configs.keys()),
            "start_time": self.start_time,
            "last_update": datetime.utcnow()
        } 