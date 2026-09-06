from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from forecasting.inference import TransformerForecastService
from forecasting.workload_forecast_provider import (
    ForecastBundle,
    ForecastProviderMode,
    ForecastRequest,
    PersistenceWorkloadForecastProvider,
    build_bundle,
    clip_nonnegative,
)


class TransformerWorkloadForecastProvider:
    mode: ForecastProviderMode = "transformer"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        dataset_root: str | Path,
        device: str = "cpu",
    ) -> None:
        self._service = TransformerForecastService.from_checkpoint(
            checkpoint,
            dataset_root=dataset_root,
            device=device,
        )
        self._fallback = PersistenceWorkloadForecastProvider()
        self.checkpoint_load_count = 1
        self.call_count = 0
        self.clip_count_by_target = np.zeros(4, dtype=np.int64)
        self.prediction_count_by_target = np.zeros(4, dtype=np.int64)
        self.fallback_count = 0

    def forecast(self, request: ForecastRequest) -> ForecastBundle:
        self.call_count += 1
        if request.workload_history is None:
            self.fallback_count += 1
            fallback = self._fallback.forecast(request)
            return ForecastBundle(
                provider=self.mode,
                issued_at_timestamp=fallback.issued_at_timestamp,
                points=fallback.points,
                forecast_history_available=False,
                persistence_fallback_used=True,
                inference_ms=0.0,
                negative_prediction_clip_count=0,
                clip_count_by_target=(0, 0, 0, 0),
            )
        history = np.asarray(request.workload_history, dtype=np.float32)
        if history.shape != (96, 8):
            raise ValueError("Transformer history must be exactly [96,8]")
        started = time.perf_counter()
        values = self._service.forecast(history)
        inference_ms = (time.perf_counter() - started) * 1000.0
        clipped, counts = clip_nonnegative(values)
        self.clip_count_by_target += np.asarray(counts, dtype=np.int64)
        self.prediction_count_by_target += clipped.shape[0]
        return build_bundle(
            provider=self.mode,
            current_timestamp=request.current_timestamp,
            values=clipped,
            history_available=True,
            fallback_used=False,
            inference_ms=inference_ms,
            clip_counts=counts,
        )

    def clip_ratios(self) -> tuple[float, float, float, float]:
        denominator = np.maximum(self.prediction_count_by_target, 1)
        return tuple(
            float(value)
            for value in self.clip_count_by_target / denominator
        )
