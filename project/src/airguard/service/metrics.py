from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._latency_count: dict[tuple[str, str], int] = defaultdict(int)
        self._prediction_count = 0

    def record_request(self, method: str, path: str, status_code: int, latency_seconds: float) -> None:
        with self._lock:
            self._request_count[(method, path, status_code)] += 1
            self._latency_sum[(method, path)] += latency_seconds
            self._latency_count[(method, path)] += 1

    def record_prediction(self) -> None:
        with self._lock:
            self._prediction_count += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP airguard_http_requests_total Total HTTP requests.",
                "# TYPE airguard_http_requests_total counter",
            ]
            for (method, path, status), value in sorted(self._request_count.items()):
                lines.append(
                    'airguard_http_requests_total{'
                    f'method="{method}",path="{path}",status="{status}"'
                    f"}} {value}"
                )

            lines.extend(
                [
                    "# HELP airguard_http_request_latency_seconds_avg Average latency by route.",
                    "# TYPE airguard_http_request_latency_seconds_avg gauge",
                ]
            )
            for (method, path), latency_sum in sorted(self._latency_sum.items()):
                count = max(self._latency_count[(method, path)], 1)
                lines.append(
                    'airguard_http_request_latency_seconds_avg{'
                    f'method="{method}",path="{path}"'
                    f"}} {latency_sum / count:.6f}"
                )

            lines.extend(
                [
                    "# HELP airguard_predictions_total Total prediction calls.",
                    "# TYPE airguard_predictions_total counter",
                    f"airguard_predictions_total {self._prediction_count}",
                ]
            )
            return "\n".join(lines) + "\n"
