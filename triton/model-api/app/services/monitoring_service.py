import logging
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.models.model_schemas import ModelPerformance

logger = logging.getLogger("monitoring-service")

class MonitoringService:
    def __init__(self, prometheus_url: str):
        """Initialize the monitoring service."""
        self.prometheus_url = prometheus_url

    async def check_health(self) -> Dict[str, Any]:
        """Check monitoring service health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prometheus_url}/-/healthy") as response:
                    if response.status == 200:
                        return {"status": "healthy", "prometheus_connected": True}
                    else:
                        return {
                            "status": "degraded",
                            "prometheus_connected": False,
                            "status_code": response.status
                        }
        except Exception as e:
            logger.error(f"Monitoring health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def get_model_performance(
        self,
        model_name: str,
        version: Optional[str] = None,
        timeframe: str = "1h"
    ) -> ModelPerformance:
        """Get performance metrics for a deployed model."""
        try:
            # Parse timeframe to determine query range
            end_time = datetime.now()
            start_time = self._parse_timeframe(end_time, timeframe)

            # Prepare model selector for Prometheus queries
            model_selector = f'model="{model_name}"'
            if version:
                model_selector += f',version="{version}"'

            # Fetch metrics from Prometheus
            throughput = await self._query_throughput(model_selector, start_time, end_time)
            latency = await self._query_latency(model_selector, start_time, end_time)
            error_rate = await self._query_error_rate(model_selector, start_time, end_time)
            gpu_utilization = await self._query_gpu_utilization(model_selector, start_time, end_time)
            memory_usage = await self._query_memory_usage(model_selector, start_time, end_time)
            request_count = await self._query_request_count(model_selector, start_time, end_time)
            avg_batch_size = await self._query_avg_batch_size(model_selector, start_time, end_time)

            # Create ModelPerformance object
            return ModelPerformance(
                model_name=model_name,
                version=version,
                timeframe=timeframe,
                throughput=throughput,
                latency=latency,
                error_rate=error_rate,
                gpu_utilization=gpu_utilization,
                memory_usage=memory_usage,
                request_count=request_count,
                average_batch_size=avg_batch_size
            )

        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            # Return empty performance data on error
            return ModelPerformance(
                model_name=model_name,
                version=version,
                timeframe=timeframe,
                throughput={},
                latency={"p50": {}, "p95": {}, "p99": {}},
                error_rate={},
                memory_usage={},
                request_count=0
            )

    def _parse_timeframe(self, end_time: datetime, timeframe: str) -> datetime:
        """Parse timeframe string and calculate start time."""
        # Parse duration format like 1h, 24h, 7d
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == 'h':
            return end_time - timedelta(hours=value)
        elif unit == 'd':
            return end_time - timedelta(days=value)
        elif unit == 'm':
            return end_time - timedelta(minutes=value)
        else:
            # Default to 1 hour
            return end_time - timedelta(hours=1)

    async def _query_prometheus(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m"
    ) -> Dict[str, Any]:
        """Query Prometheus for metrics data."""
        try:
            params = {
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": step
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prometheus_url}/api/v1/query_range", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"Prometheus query failed: {response.status}")
                        return {"status": "error", "data": {"result": []}}

        except Exception as e:
            logger.error(f"Error querying Prometheus: {str(e)}")
            return {"status": "error", "data": {"result": []}}

    async def _query_throughput(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """Query model throughput from Prometheus."""
        query = f'rate(triton_inference_count{{{model_selector}}}[1m])'
        result = await self._query_prometheus(query, start_time, end_time)

        # Process result to a time series
        throughput = {}
        if result.get("status") == "success":
            for series in result["data"]["result"]:
                for timestamp, value in series.get("values", []):
                    throughput[timestamp] = float(value)

        return throughput

    async def _query_latency(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Query model latency from Prometheus."""
        # Query for different percentiles
        percentiles = {
            "p50": 0.5,
            "p95": 0.95,
            "p99": 0.99
        }

        latency = {"p50": {}, "p95": {}, "p99": {}}

        for name, value in percentiles.items():
            query = f'histogram_quantile({value}, sum(rate(triton_inference_latency_bucket{{{model_selector}}}[1m])) by (le))'
            result = await self._query_prometheus(query, start_time, end_time)

            if result.get("status") == "success":
                for series in result["data"]["result"]:
                    for timestamp, value in series.get("values", []):
                        latency[name][timestamp] = float(value) * 1000  # Convert to ms

        return latency

    async def _query_error_rate(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """Query model error rate from Prometheus."""
        query = f'sum(rate(triton_inference_error_count{{{model_selector}}}[1m])) / sum(rate(triton_inference_count{{{model_selector}}}[1m]))'
        result = await self._query_prometheus(query, start_time, end_time)

        # Process result to a time series
        error_rate = {}
        if result.get("status") == "success":
            for series in result["data"]["result"]:
                for timestamp, value in series.get("values", []):
                    error_rate[timestamp] = float(value) if value != "NaN" else 0.0

        return error_rate

    async def _query_gpu_utilization(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[Dict[str, float]]:
        """Query GPU utilization from Prometheus."""
        # This query assumes DCGM metrics are available
        query = f'DCGM_FI_PROF_GR_ENGINE_ACTIVE{{{model_selector}}}'
        result = await self._query_prometheus(query, start_time, end_time)

        # Process result to a time series
        gpu_utilization = {}
        if result.get("status") == "success" and result["data"]["result"]:
            for series in result["data"]["result"]:
                for timestamp, value in series.get("values", []):
                    gpu_utilization[timestamp] = float(value) if value != "NaN" else 0.0

            return gpu_utilization

        return None  # GPU metrics not available

    async def _query_memory_usage(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """Query memory usage from Prometheus."""
        query = f'container_memory_usage_bytes{{pod=~"triton.*", {model_selector}}}'
        result = await self._query_prometheus(query, start_time, end_time)

        # Process result to a time series
        memory_usage = {}
        if result.get("status") == "success":
            for series in result["data"]["result"]:
                for timestamp, value in series.get("values", []):
                    memory_usage[timestamp] = float(value) / (1024 * 1024)  # Convert to MB

        return memory_usage

    async def _query_request_count(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Query total request count from Prometheus."""
        query = f'sum(increase(triton_inference_count{{{model_selector}}}[{int((end_time - start_time).total_seconds())}s]))'

        # This is a single value query, so use different endpoint
        async with aiohttp.ClientSession() as session:
            params = {"query": query}
            async with session.get(f"{self.prometheus_url}/api/v1/query", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data["data"]["result"]:
                        return int(float(data["data"]["result"][0]["value"][1]))

        return 0

    async def _query_avg_batch_size(
        self,
        model_selector: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        """Query average batch size from Prometheus."""
        query = f'avg(avg_over_time(triton_inference_batch_size{{{model_selector}}}[{int((end_time - start_time).total_seconds())}s]))'

        async with aiohttp.ClientSession() as session:
            params = {"query": query}
            async with session.get(f"{self.prometheus_url}/api/v1/query", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data["data"]["result"]:
                        return float(data["data"]["result"][0]["value"][1])

        return None
