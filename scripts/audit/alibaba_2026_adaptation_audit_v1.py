from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "alibaba_2026_adaptation_audit_v1"
RAW_ROOT = ROOT / "data" / "raw" / "alibaba_2026"
SPOT_ROOT = RAW_ROOT / "spot_gpu"
DOC_ROOT = RAW_ROOT / "documentation"
ALIBABA_2020 = (
    ROOT
    / "references"
    / "external_repos"
    / "sustain-cluster"
    / "data"
    / "workload"
    / "alibaba_2020_dataset"
    / "result_df_full_year_2020.pkl"
)
SUSTAINCLUSTER_ROOT = ROOT / "references" / "external_repos" / "sustain-cluster"

ALIBABA_2020_SIZE = 142_336_499
ALIBABA_2020_SHA256 = (
    "3BA8A8E0067288F8D2542752A292F7DA04CB79305689BA145CC6060E96CB7F2E"
)

AUDIT_DATE = "2026-09-04"
UPSTREAM_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
PROJECT_HEAD = "90eb972f78742603b522fabc3e1cf65391434418"
SUSTAINCLUSTER_HEAD = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"

GPU2026_README = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/README.md"
GPU2026_SCHEMA = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"
GPU2026_DOWNLOAD = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md"
SPOT2026_README = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2026-spot-gpu/README.md"
ROOT_README = "https://github.com/alibaba/clusterdata/blob/master/README.md"

EXPECTED_FILES = {
    SPOT_ROOT / "job_info_df.csv": (
        23_055_929,
        "113CCEE4C28F5C3BBAACA974CD164B9280B7D4C39E53B745443B28EEA05E03DD",
    ),
    SPOT_ROOT / "node_info_df.csv": (
        78_330,
        "1ABA161961A5A4A1A61AA581383C5E5ABE3400B59F8597BA8C4EEF7597BC9D18",
    ),
    DOC_ROOT / "gpu2026_README.md": (
        3_377,
        "D8E4878B6D341486FD9700EA668E85B4BCF3F82F56D6C426B14C347324002CC4",
    ),
    DOC_ROOT / "gpu2026_schema.md": (
        7_694,
        "6E8A6C5B2487E89BEC25D59D052191B86B0E585FD31F16DA227C2928BA2E4A10",
    ),
    DOC_ROOT / "gpu2026_data_download.md": (
        1_939,
        "69E3DB5F6219EB9C457D4F3718D203F974027A2AD96D68F6EF4D446ECE960704",
    ),
    DOC_ROOT / "spotgpu2026_README.md": (
        5_608,
        "11EB33818D3B40BA430C1BC1E0FD42786AF214DB334152141E29BCCB3D26DB2B",
    ),
    DOC_ROOT / "repository_metadata.json": (
        6_278,
        "8405C46F1BAF82AFC9181C5E5F81E7B537C1E9E462E7589102E479FFB31CDBD5",
    ),
    DOC_ROOT / "master_commit.json": (
        159_047,
        "4F0469526700A232F5E92BD203AFD453B8D8707587686AEBEA29AE740BDE16F4",
    ),
}

SPOT_JOB_COLUMNS = (
    "job_name",
    "organization",
    "gpu_model",
    "cpu_request",
    "gpu_request",
    "worker_num",
    "submit_time",
    "duration",
    "job_type",
)
SPOT_NODE_COLUMNS = (
    "gpu_model",
    "gpu_capacity_num",
    "cpu_num",
    "node_name",
)

CANONICAL_COLUMNS = (
    "source",
    "source_record_id",
    "sample_evidence",
    "task_id",
    "arrival_time",
    "arrival_time_unit",
    "cpu_request",
    "gpu_request",
    "memory_request",
    "bandwidth",
    "true_duration",
    "duration_unit",
    "estimated_duration",
    "priority",
    "task_type",
    "gpu_type",
    "worker_num",
    "origin_dc",
    "current_dc",
    "sla",
    "resource_scope",
    "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def validate_frozen_workspace() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    sustain_head = git("rev-parse", "HEAD", cwd=SUSTAINCLUSTER_ROOT)
    sustain_status = git("status", "--short", cwd=SUSTAINCLUSTER_ROOT)
    if branch != "feature/baseline-repair-information-contract-v1":
        raise RuntimeError(f"unexpected branch: {branch}")
    if head != PROJECT_HEAD:
        raise RuntimeError(f"unexpected project HEAD: {head}")
    if sustain_head != SUSTAINCLUSTER_HEAD or sustain_status:
        raise RuntimeError("SustainCluster reference is not at the frozen clean commit")
    return {
        "branch": branch,
        "head": head,
        "sustaincluster_head": sustain_head,
        "dirty_workspace_preserved": bool(git("status", "--short")),
    }


def validate_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, (size, expected_hash) in EXPECTED_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing audited source snapshot: {path}")
        if path.stat().st_size != size or sha256(path) != expected_hash:
            raise RuntimeError(f"source integrity mismatch: {path}")
    if not ALIBABA_2020.is_file():
        raise FileNotFoundError(f"missing frozen Alibaba2020 baseline: {ALIBABA_2020}")
    if (
        ALIBABA_2020.stat().st_size != ALIBABA_2020_SIZE
        or sha256(ALIBABA_2020) != ALIBABA_2020_SHA256
    ):
        raise RuntimeError("frozen Alibaba2020 baseline integrity mismatch")
    alibaba_2020 = pd.read_pickle(ALIBABA_2020)
    if alibaba_2020.shape != (37_552, 2) or tuple(alibaba_2020.columns) != (
        "interval_15m",
        "tasks_matrix",
    ):
        raise RuntimeError("frozen Alibaba2020 processed schema changed")
    jobs = pd.read_csv(SPOT_ROOT / "job_info_df.csv")
    nodes = pd.read_csv(SPOT_ROOT / "node_info_df.csv")
    if tuple(jobs.columns) != SPOT_JOB_COLUMNS or tuple(nodes.columns) != SPOT_NODE_COLUMNS:
        raise RuntimeError("SpotGPU source schema changed")
    if jobs.shape != (466_867, 9) or nodes.shape != (4_278, 4):
        raise RuntimeError("SpotGPU source row count changed")
    if jobs.isna().any().any() or nodes.isna().any().any():
        raise RuntimeError("SpotGPU source unexpectedly contains nulls")
    if not jobs["submit_time"].is_monotonic_increasing:
        raise RuntimeError("SpotGPU submit_time is not monotonic in file order")
    if jobs["job_name"].duplicated().any():
        raise RuntimeError("SpotGPU job_name is not unique")
    return jobs, nodes


def markdown_table(frame: pd.DataFrame) -> str:
    def value(item: Any) -> str:
        if pd.isna(item):
            return "MISSING"
        return str(item).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(value(item) for item in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def target_schema() -> pd.DataFrame:
    rows = [
        ("task", "task_id", "Required", "Task.job_name / TaskSpec.task_id", "all", "必须唯一；源 ID 需保留来源前缀"),
        ("task", "arrival_time", "Required", "Task.arrival_time", "env arrival", "当前主线按 15 分钟调度边界注入"),
        ("task", "cpu_request", "Required", "Task.cores_req", "capacity constraint", "非负 CPU core"),
        ("task", "gpu_request", "Required", "Task.gpu_req", "capacity constraint", "当前仅表达 GPU-equivalent 数量"),
        ("task", "memory_request", "Required", "Task.mem_req", "capacity constraint", "必须是主机内存，不是 GPU 显存"),
        ("task", "bandwidth", "Required", "Task.bandwidth_gb", "transfer cost/delay", "当前语义为需传输数据量 GB，不是链路速率"),
        ("task", "estimated_duration", "Required", "task.estimated_duration", "deployable MPC", "只能由提交时可知信息估计"),
        ("task", "true_duration", "Simulator-only", "task.true_duration", "runtime ground truth", "不可进入在线 observation/MPC"),
        ("task", "origin_dc", "Synthetic", "Task.origin_dc_id", "network mapping", "公开 trace 无多 DC 地理来源，需场景模型"),
        ("task", "current_dc", "Simulator-only", "Task.dest_dc_id/runtime state", "running state", "由 simulator 调度结果产生"),
        ("task", "sla", "Synthetic", "Task.sla_deadline", "deadline/penalty", "当前默认 duration multiplier；2026 类型可用于后续分层"),
        ("task", "priority", "Optional", "TaskSpec.priority; SustainCluster Task 不原生消费", "scenario metadata", "不可直接改 waiting penalty"),
        ("task", "task_type", "Optional", "not represented", "future scenario metadata", "GPU2026 提供，SpotGPU2026 不提供 workload type"),
        ("task", "gpu_type", "Optional", "not represented", "future compatibility", "当前 simulator 不支持型号约束"),
        ("task", "worker_num", "Optional", "not represented", "source granularity", "需先确认 request 是每 worker 还是整 job"),
        ("system", "dc_capacity", "Required", "datacenters.yaml", "capacity constraints", "五 DC 场景配置"),
        ("system", "gpu_type_capacity", "Optional", "not represented", "future compatibility", "当前仅 total_gpus"),
        ("system", "cpu_capacity", "Required", "DataCenter.total_cores", "capacity constraints", "场景配置"),
        ("system", "memory_capacity", "Required", "DataCenter.total_mem", "capacity constraints", "两套 2026 均缺主机内存容量"),
        ("system", "network", "Required", "network cost matrix", "transfer objective/delay", "trace 网络流量不能直接替代跨 DC 成本"),
        ("system", "topology", "Optional", "not represented beyond DC", "future locality", "GPU2026 提供 cluster/ASW/server 层级"),
        ("system", "electricity", "Required", "external price provider", "MPC objective", "由项目能源数据提供，不来自 workload trace"),
        ("system", "carbon", "Required", "external carbon provider", "MPC objective", "由项目碳数据提供，不来自 workload trace"),
    ]
    return pd.DataFrame(
        rows,
        columns=("scope", "target_field", "classification", "current_code_source", "consumer", "notes"),
    )


def source_mapping() -> pd.DataFrame:
    rows = [
        ("task_id", "job_name", "DIRECT", "pod_id/workload_id", "DIRECT", "job_name", "DIRECT", "2020 task; GPU2026 pod; Spot job", "HIGH", "粒度不同"),
        ("arrival_time", "interval_15m", "DERIVED", "day+hour observation", "MISSING", "submit_time", "DIRECT", "15 min / 1 hour / relative seconds", "HIGH", "GPU2026 小时观测不是 arrival"),
        ("cpu_request", "cpu_usage scaled", "DERIVED", "cpu_request_cores", "DIRECT", "cpu_request", "DIRECT", "task/lifetime vs hourly vs job", "MEDIUM", "Spot request 范围未明确到 worker/job"),
        ("gpu_request", "gpu_wrk_util scaled", "DERIVED", "gpu_request", "DIRECT", "gpu_request", "DIRECT", "task/lifetime vs hourly vs job", "MEDIUM", "GPU-equivalent 可保留小数"),
        ("memory_request", "avg_mem scaled", "DERIVED", "gpu_mem_request + avg_memory_util", "MISSING", "none", "MISSING", "task lifetime / hourly / none", "HIGH", "GPU 显存不能替代主机内存"),
        ("bandwidth", "bandwidth_gb processed", "DERIVED", "server rx/tx hourly", "MISSING", "none", "MISSING", "task lifetime / server-hour / none", "HIGH", "GPU2026 网络不能精确归因到 task"),
        ("true_duration", "duration_min", "DERIVED", "duration_hours via pod_id", "DERIVED", "duration", "DIRECT", "minutes / hours / seconds", "HIGH", "GPU2026 需 join；均只能作 simulator ground truth"),
        ("estimated_duration", "runtime estimator", "MODELED", "none", "MODELED", "none", "MODELED", "submission-time", "HIGH", "必须仅用历史 train 数据拟合"),
        ("origin_dc", "population/time model", "MODELED", "none", "MODELED", "none", "MODELED", "scenario", "HIGH", "三套 trace 均不提供跨 DC 来源"),
        ("current_dc", "simulator state", "SIMULATOR", "server_id only", "PARTIAL", "none", "MISSING", "runtime", "HIGH", "源 server 不等于项目五 DC"),
        ("sla", "1.5 x duration", "MODELED", "priority/job type", "MODELED", "HP/Spot", "MODELED", "scenario", "MEDIUM", "类别可指导但不足以给出数值 deadline"),
        ("priority", "none", "MISSING", "priority_class", "DIRECT", "job_type", "DIRECT", "hourly / job", "HIGH", "Spot job_type 实际是 HP/Spot 优先级类"),
        ("task_type", "limited workload tags", "PARTIAL", "job_type_public", "DIRECT", "none", "MISSING", "hourly", "HIGH", "Spot 不提供 training/inference 分类"),
        ("gpu_type", "gpu_type/gpu_type_spec", "DIRECT", "gpu_spec_public", "DIRECT", "gpu_model", "DIRECT", "task/hour/job", "HIGH", "当前 simulator 尚不消费"),
        ("worker_num", "inst_num", "DIRECT", "pods per workload derivable", "DERIVED", "worker_num", "DIRECT", "task/workload/job", "MEDIUM", "聚合资源前需锁定 scope"),
        ("node_cpu_capacity", "pai_machine_spec.cap_cpu", "DIRECT", "server_cpu_capacity_cores", "DIRECT", "cpu_num", "DIRECT", "machine/hour/node", "HIGH", "可用于容量分布，不直接替换五 DC 配置"),
        ("node_gpu_capacity", "pai_machine_spec.cap_gpu", "DIRECT", "server_gpu_count/gpu_count", "DIRECT", "gpu_capacity_num", "DIRECT", "machine/hour/node", "HIGH", "支持异构容量统计"),
        ("node_memory_capacity", "pai_machine_spec.cap_mem", "DIRECT", "none", "MISSING", "none", "MISSING", "machine", "HIGH", "2026 两源均缺"),
        ("topology", "machine only", "PARTIAL", "cluster_id/asw_id/server_id", "DIRECT", "node only", "POOR", "hourly/static", "HIGH", "GPU2026 最强"),
        ("network", "sensor read/write", "PARTIAL", "rx_gibps_avg/tx_gibps_avg", "PARTIAL", "none", "MISSING", "lifetime/hourly/none", "HIGH", "都不是跨 DC 传输成本"),
        ("electricity/carbon", "none", "MISSING", "none", "MISSING", "none", "MISSING", "external", "HIGH", "继续由项目外部能源时间序列提供"),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "target_field",
            "alibaba2020_source",
            "alibaba2020_mapping",
            "gpu2026_source",
            "gpu2026_mapping",
            "spotgpu2026_source",
            "spotgpu2026_mapping",
            "time_resolution",
            "reliability",
            "notes",
        ),
    )


def gpu_heterogeneity(jobs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    node_stats = nodes.groupby("gpu_model", sort=True).agg(
        node_count=("node_name", "count"),
        gpu_capacity=("gpu_capacity_num", "sum"),
        cpu_capacity=("cpu_num", "sum"),
    )
    job_values = jobs.assign(
        gpu_request_x_worker_diagnostic=jobs["gpu_request"] * jobs["worker_num"]
    ).groupby("gpu_model", sort=True).agg(
        job_count=("job_name", "count"),
        requested_gpu_field_sum=("gpu_request", "sum"),
        requested_gpu_x_worker_diagnostic=("gpu_request_x_worker_diagnostic", "sum"),
    )
    spot = node_stats.join(job_values, how="outer").reset_index()
    spot.insert(0, "evidence_level", "LOCAL_FULL_FILE")
    spot.insert(0, "source", "SpotGPU2026")
    spot["notes"] = "x_worker 仅诊断，不认定官方 request 粒度"
    gpu = pd.DataFrame(
        [
            {
                "source": "GPU2026",
                "evidence_level": "DOCUMENTATION_LEVEL_ONLY",
                "gpu_model": "ALL_PUBLIC_BUCKETS",
                "node_count": 37_707,
                "gpu_capacity": 155_410,
                "cpu_capacity": "MISSING",
                "job_count": "MISSING",
                "requested_gpu_field_sum": "MISSING",
                "requested_gpu_x_worker_diagnostic": "MISSING",
                "notes": "node/GPU 为官方 hourly peak，不是分型号本地统计",
            }
        ]
    )
    return pd.concat([gpu, spot], ignore_index=True)


def priority_task_type(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in ("HP", "LP", "Other"):
        rows.append(
            {
                "source": "GPU2026",
                "dimension": "priority_class",
                "value": value,
                "count": "NOT LOCALLY COUNTED",
                "evidence_level": "DOCUMENTATION_LEVEL_ONLY",
                "semantics": {
                    "HP": "guaranteed high priority",
                    "LP": "low-priority/spot flexible capacity",
                    "Other": "outside HP/LP bands",
                }[value],
            }
        )
    for value in (
        "training",
        "online_inference",
        "offline_inference",
        "dev",
        "other",
        "unknown",
    ):
        rows.append(
            {
                "source": "GPU2026",
                "dimension": "task_type",
                "value": value,
                "count": "NOT LOCALLY COUNTED",
                "evidence_level": "DOCUMENTATION_LEVEL_ONLY",
                "semantics": "official public job_type bucket",
            }
        )
    for value, count in jobs["job_type"].value_counts().items():
        rows.append(
            {
                "source": "SpotGPU2026",
                "dimension": "priority_class",
                "value": value,
                "count": int(count),
                "evidence_level": "LOCAL_FULL_FILE",
                "semantics": "strict-SLO HP" if value == "HP" else "opportunistic spot-resource job",
            }
        )
    rows.append(
        {
            "source": "SpotGPU2026",
            "dimension": "task_type",
            "value": "MISSING",
            "count": len(jobs),
            "evidence_level": "LOCAL_FULL_FILE",
            "semantics": "job_type is HP/Spot, not training/inference type",
        }
    )
    return pd.DataFrame(rows)


def time_resolution() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Alibaba2020 current project", "raw relative seconds; processed 15 min", "15 min in frozen pickle", "15 min task batches", "minutes", "15 min", "YES", "主线保持不变"),
            ("Alibaba GPU2026", "relative day + hour", "NO exact task arrival", "hourly pod/server/network", "duration_hours summary", "1 hour", "NO", "仅小时外部/场景验证；小时内若生成必须标 SYNTHETIC"),
            ("Alibaba SpotGPU2026", "relative seconds", "submit_time seconds", "event-level jobs; static nodes", "duration seconds", "event-level; adapter 可 bin 15 min", "YES AFTER BINNING", "保留原始秒并记录 bin 规则"),
        ],
        columns=(
            "source",
            "native_time_unit",
            "arrival_resolution",
            "state_resolution",
            "duration_resolution",
            "recommended_simulator_timestep",
            "can_support_15min_directly",
            "notes",
        ),
    )


def gpu2026_canonical_template() -> pd.DataFrame:
    row = {column: "MISSING" for column in CANONICAL_COLUMNS}
    row.update(
        {
            "source": "Alibaba GPU2026",
            "source_record_id": "NO_LOCAL_FACT_ROW",
            "sample_evidence": "DOCUMENTATION_LEVEL_ONLY",
            "arrival_time_unit": "relative day/hour observation only",
            "duration_unit": "hours when joined from execution summary",
            "resource_scope": "pod-hour schema",
            "notes": "351.8 GB pod archive intentionally not downloaded; this row is a non-fabricated schema template, not a trace record",
        }
    )
    return pd.DataFrame([row], columns=CANONICAL_COLUMNS)


def spot_canonical_sample(jobs: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    rows = []
    for item in jobs.head(count).itertuples(index=False):
        rows.append(
            {
                "source": "Alibaba SpotGPU2026",
                "source_record_id": str(int(item.job_name)),
                "sample_evidence": "LOCAL_FULL_FILE_HEAD",
                "task_id": f"spotgpu2026:{int(item.job_name)}",
                "arrival_time": f"T+{int(item.submit_time)}s",
                "arrival_time_unit": "seconds since first submitted job",
                "cpu_request": float(item.cpu_request),
                "gpu_request": float(item.gpu_request),
                "memory_request": "MISSING",
                "bandwidth": "MISSING",
                "true_duration": int(item.duration),
                "duration_unit": "seconds",
                "estimated_duration": "MISSING",
                "priority": str(item.job_type),
                "task_type": "MISSING",
                "gpu_type": str(item.gpu_model),
                "worker_num": int(item.worker_num),
                "origin_dc": "MISSING",
                "current_dc": "MISSING",
                "sla": "MISSING",
                "resource_scope": "OFFICIAL DOC DOES NOT DISAMBIGUATE PER-WORKER VS PER-JOB",
                "notes": "No multiplication by worker_num; no modeled fields injected",
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def canonical_provenance() -> pd.DataFrame:
    rows = []
    specifications = {
        "GPU2026": {
            "task_id": ("pod_id", "identity with source prefix", "DIRECT"),
            "arrival_time": ("day/hour", "exact arrival unavailable", "MISSING"),
            "cpu_request": ("cpu_request_cores", "identity", "DIRECT"),
            "gpu_request": ("gpu_request", "identity GPU-equivalent", "DIRECT"),
            "memory_request": ("none (gpu_mem_request is accelerator memory)", "none", "MISSING"),
            "bandwidth": ("none (server rx/tx is aggregate)", "none", "MISSING"),
            "true_duration": ("duration_hours joined by pod_id", "hours to simulator units", "DERIVED"),
            "estimated_duration": ("historical train-only estimator", "future design", "MODELED"),
            "priority": ("priority_class", "identity", "DIRECT"),
            "task_type": ("job_type_public", "identity", "DIRECT"),
            "gpu_type": ("gpu_spec_public", "identity", "DIRECT"),
            "worker_num": ("pods per workload", "group count if workload_id present", "DERIVED"),
            "origin_dc": ("none", "scenario assignment", "MODELED"),
            "current_dc": ("simulator result", "runtime", "MODELED"),
            "sla": ("priority/task type", "scenario policy", "MODELED"),
        },
        "SpotGPU2026": {
            "task_id": ("job_name", "prefix source", "DIRECT"),
            "arrival_time": ("submit_time", "T + relative seconds", "DERIVED"),
            "cpu_request": ("cpu_request", "identity; scope unresolved", "DIRECT"),
            "gpu_request": ("gpu_request", "identity; scope unresolved", "DIRECT"),
            "memory_request": ("none", "none", "MISSING"),
            "bandwidth": ("none", "none", "MISSING"),
            "true_duration": ("duration", "seconds to simulator units", "DIRECT"),
            "estimated_duration": ("historical train-only estimator", "future design", "MODELED"),
            "priority": ("job_type", "HP/Spot", "DIRECT"),
            "task_type": ("none", "none", "MISSING"),
            "gpu_type": ("gpu_model", "identity", "DIRECT"),
            "worker_num": ("worker_num", "identity; do not multiply before scope audit", "DIRECT"),
            "origin_dc": ("none", "scenario assignment", "MODELED"),
            "current_dc": ("simulator result", "runtime", "MODELED"),
            "sla": ("job_type", "class-conditioned scenario policy", "MODELED"),
        },
    }
    for source, fields in specifications.items():
        for field, (source_field, transform, provenance) in fields.items():
            rows.append(
                {
                    "source_dataset": source,
                    "canonical_field": field,
                    "source_field": source_field,
                    "transformation": transform,
                    "provenance_type": provenance,
                    "evidence_level": "DOCUMENTATION_LEVEL_ONLY" if source == "GPU2026" else "LOCAL_FULL_FILE",
                }
            )
    return pd.DataFrame(rows)


def compatibility_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("task-level arrival", "GOOD", "POOR", "GOOD", "GPU2026 只有 pod-hour 观测；Spot 有相对秒 submit_time"),
            ("duration", "GOOD", "PARTIAL", "GOOD", "GPU2026 可按 pod_id 连接 execution summary；均只作 true duration"),
            ("CPU", "PARTIAL", "GOOD", "GOOD", "2020 当前处理取 usage；2026 为 request"),
            ("GPU", "PARTIAL", "GOOD", "GOOD", "2020 当前处理取 utilization；2026 为 request"),
            ("memory", "PARTIAL", "MISSING", "MISSING", "GPU2026 gpu_mem_request 是显存，不是 host memory"),
            ("bandwidth", "PARTIAL", "POOR", "MISSING", "GPU2026 只有 server-hour rx/tx，不能作为 task GB"),
            ("priority", "POOR", "GOOD", "GOOD", "GPU2026 HP/LP/Other；Spot HP/Spot"),
            ("task type", "PARTIAL", "GOOD", "MISSING", "Spot job_type 是优先级，不是 workload type"),
            ("GPU heterogeneity", "PARTIAL", "GOOD", "GOOD", "当前 simulator 仍需兼容层或扩展"),
            ("topology", "PARTIAL", "GOOD", "POOR", "GPU2026 有 cluster/ASW/server；Spot 仅 node"),
            ("network", "PARTIAL", "PARTIAL", "MISSING", "无一直接提供跨 DC cost/delay"),
            ("15min compatibility", "GOOD", "POOR", "GOOD", "Spot event 可确定性分箱；GPU2026 不可恢复小时内真实到达"),
            ("information-contract compatibility", "GOOD", "POOR", "PARTIAL", "Spot 尚缺 memory/bandwidth/origin/SLA/estimated duration"),
        ],
        columns=("dimension", "alibaba2020", "gpu2026", "spotgpu2026", "reason"),
    )


def v3_schema() -> pd.DataFrame:
    rows = [
        ("state", "scenario_id", "string", "DEPLOYABLE_METADATA", "scenario construction", "required"),
        ("state", "episode_id", "string", "DEPLOYABLE_METADATA", "generator", "required"),
        ("state", "step", "int", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "timestamp", "datetime/relative", "DEPLOYABLE_CURRENT", "source+scenario", "required"),
        ("state", "pending_tasks", "nested task refs", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "running_tasks", "nested task refs", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "in_transit_tasks", "nested task refs", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "dc_states", "nested DC snapshots", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "price", "float[DC]", "DEPLOYABLE_CURRENT", "energy provider", "required"),
        ("state", "carbon", "float[DC]", "DEPLOYABLE_CURRENT", "carbon provider", "required"),
        ("state", "reservations", "resource[DC]", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("state", "current_risk", "float", "EVALUATION_METADATA", "risk diagnostic", "optional"),
        ("task", "task_id", "string", "DEPLOYABLE_CURRENT", "source", "required"),
        ("task", "original_index", "int", "DEPLOYABLE_CURRENT", "scheduler order", "required"),
        ("task", "cpu", "float", "DEPLOYABLE_CURRENT", "source/direct or modeled", "required"),
        ("task", "gpu", "float", "DEPLOYABLE_CURRENT", "source/direct or modeled", "required"),
        ("task", "memory", "float", "DEPLOYABLE_CURRENT", "source or explicit model", "required"),
        ("task", "bandwidth", "float", "DEPLOYABLE_CURRENT", "source or explicit model", "required"),
        ("task", "estimated_duration", "float", "DEPLOYABLE_CURRENT", "train-only estimator", "required"),
        ("task", "true_duration", "float", "SIMULATOR_GROUND_TRUTH", "observed duration", "required"),
        ("task", "origin", "int", "DEPLOYABLE_CURRENT", "scenario model", "required"),
        ("task", "current_location", "int/null", "DEPLOYABLE_CURRENT", "simulator", "required"),
        ("task", "priority", "string", "DEPLOYABLE_CURRENT", "source/mapping", "optional"),
        ("task", "task_type", "string", "DEPLOYABLE_CURRENT", "source", "optional"),
        ("task", "gpu_type", "string", "DEPLOYABLE_CURRENT", "source", "optional"),
        ("task", "worker_num", "int", "DEPLOYABLE_CURRENT", "source", "optional"),
        ("task", "sla", "datetime/relative", "DEPLOYABLE_CURRENT", "source class + scenario model", "required"),
        ("task", "feasible_mask", "bool[6]", "DEPLOYABLE_CURRENT", "adapter", "required"),
        ("task", "h1_action", "int", "LABEL_ONLY", "H1 MPC", "required"),
        ("task", "oracle_mpc_action", "int", "TEACHER_ONLY", "H4/oracle MPC", "required"),
        ("privileged_future", "future_arrivals", "nested", "PRIVILEGED_FUTURE", "scenario future", "separate directory"),
        ("privileged_future", "future_price", "float[H,DC]", "PRIVILEGED_FUTURE", "energy trace", "separate directory"),
        ("privileged_future", "future_carbon", "float[H,DC]", "PRIVILEGED_FUTURE", "carbon trace", "separate directory"),
    ]
    return pd.DataFrame(
        rows,
        columns=("level", "field", "dtype", "information_class", "source", "requirement"),
    )


def raw_manifest() -> pd.DataFrame:
    urls = {
        "job_info_df.csv": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-spot-gpu/job_info_df.csv",
        "node_info_df.csv": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-spot-gpu/node_info_df.csv",
        "gpu2026_README.md": GPU2026_README,
        "gpu2026_schema.md": GPU2026_SCHEMA,
        "gpu2026_data_download.md": GPU2026_DOWNLOAD,
        "spotgpu2026_README.md": SPOT2026_README,
        "repository_metadata.json": "https://api.github.com/repos/alibaba/clusterdata",
        "master_commit.json": "https://api.github.com/repos/alibaba/clusterdata/commits/master",
    }
    rows = []
    for path, (size, expected_hash) in EXPECTED_FILES.items():
        if path.parent == SPOT_ROOT or path.name == "spotgpu2026_README.md":
            source_dataset = "SpotGPU2026"
        elif path.name in {"repository_metadata.json", "master_commit.json"}:
            source_dataset = "Alibaba clusterdata repository"
        else:
            source_dataset = "GPU2026"
        status = (
            "FULL_LOCAL_SOURCE_DATA"
            if path.suffix == ".csv"
            else "UPSTREAM_REPOSITORY_API_SNAPSHOT"
            if path.suffix == ".json"
            else "OFFICIAL_DOCUMENTATION_SNAPSHOT"
        )
        rows.append(
            {
                "source_dataset": source_dataset,
                "file": path.name,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "url": urls[path.name],
                "download_date": AUDIT_DATE,
                "file_size_bytes": size,
                "sha256": expected_hash,
                "status": status,
                "git_ignored": "YES",
            }
        )
    archives = (
        ("asi_opensource_pod_hourly.zip", 351_803_513_445),
        ("asi_opensource_server_hourly.zip", 3_080_121_387),
        ("asi_opensource_network_hourly.zip", 203_904_100),
        ("asi_opensource_job_execution_summary.zip", 1_188_295_031),
    )
    for name, size in archives:
        rows.append(
            {
                "source_dataset": "GPU2026",
                "file": name,
                "local_path": "NOT_DOWNLOADED",
                "url": f"https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/{name}",
                "download_date": "NOT_DOWNLOADED",
                "file_size_bytes": size,
                "sha256": "NOT PROVIDED IN OFFICIAL DOWNLOAD DOC",
                "status": "NOT_DOWNLOADED_INTENTIONALLY",
                "git_ignored": "N/A",
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(output: Path, jobs: pd.DataFrame, heterogeneity: pd.DataFrame) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    spot = heterogeneity.loc[heterogeneity["source"].eq("SpotGPU2026")].copy()
    spot[["gpu_model", "node_count", "gpu_capacity"]].to_csv(
        plot_dir / "01_gpu_heterogeneity_source.csv", index=False
    )
    x = np.arange(len(spot))
    width = 0.38
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, spot["node_count"].astype(float), width, label="Nodes", color="#277DA1")
    axis.bar(x + width / 2, spot["gpu_capacity"].astype(float), width, label="GPU capacity", color="#43AA8B")
    axis.set_xticks(x, spot["gpu_model"], rotation=25, ha="right")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_spotgpu_gpu_heterogeneity.png", dpi=160)
    plt.close(fig)

    priority = jobs["job_type"].value_counts().rename_axis("job_type").reset_index(name="job_count")
    priority.to_csv(plot_dir / "02_priority_source.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(priority["job_type"], priority["job_count"], color=["#4D908E", "#F9844A"])
    axis.set_ylabel("Jobs")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_spotgpu_priority_counts.png", dpi=160)
    plt.close(fig)

    weekly = (
        jobs.assign(trace_week=(jobs["submit_time"] // (7 * 86_400)).astype(int))
        .groupby("trace_week", as_index=False)
        .agg(job_count=("job_name", "count"))
    )
    weekly.to_csv(plot_dir / "03_arrival_timeline_source.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(weekly["trace_week"], weekly["job_count"], color="#577590", linewidth=2)
    axis.set_xlabel("Relative trace week")
    axis.set_ylabel("Submitted jobs")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_spotgpu_weekly_arrivals.png", dpi=160)
    plt.close(fig)
    write_text(
        plot_dir / "README.md",
        "# Plot Sources\n\nEach PNG has a same-numbered CSV source file. All three plots use the locally verified full SpotGPU2026 CSVs; no GPU2026 fact-table counts are plotted because those archives were not downloaded.",
    )


def write_reports(
    output: Path,
    jobs: pd.DataFrame,
    nodes: pd.DataFrame,
    target: pd.DataFrame,
    mapping: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    priority: pd.DataFrame,
    time_table: pd.DataFrame,
    gpu_sample: pd.DataFrame,
    spot_sample: pd.DataFrame,
    provenance: pd.DataFrame,
    compatibility: pd.DataFrame,
    schema: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    target.to_csv(output / "01_target_task_schema.csv", index=False)
    write_text(
        output / "02_target_task_schema.md",
        "# 当前 SustainCluster 目标任务契约\n\n"
        "审计来源为本地 `Task`、workload extractor、state/horizon adapter、MPC 与 StructuredCurrent 实现。`priority/task_type/gpu_type/worker_num` 尚未进入当前 SustainCluster 决策语义。\n\n"
        + markdown_table(target),
    )
    mapping.to_csv(output / "03_source_field_mapping.csv", index=False)
    write_text(
        output / "04_source_field_mapping.md",
        "# 三源字段映射\n\n"
        "`DIRECT` 仅表示源字段直接存在，不代表已经满足当前 simulator 的单位、粒度或语义。GPU2026 为文档级审计；SpotGPU2026 为本地完整文件统计；Alibaba2020 为当前冻结 pickle 与代码链审计。\n\n"
        + markdown_table(mapping)
        + """

## Memory / Bandwidth 缺失字段处理

1. **不用于完整任务模型**：来源最保守，但无法验证当前包含 host memory 与跨 DC transfer 的完整信息契约。
2. **显式合成模型**：为 memory、bandwidth 设置带版本号的条件分布，并做参数敏感性分析；可以形成完整 scenario，但结论只能归因于“真实 arrival/resource + modeled missing fields”，不能归因于原始 trace。
3. **CPU/GPU 子问题验证**：只验证事件到达、duration、CPU/GPU request、priority 与 GPU type；模型风险最低，但不能得出热内存压力或网络迁移收益结论。

当前最小适配推荐方案 3。若进入 v3 Scenario B，推荐方案 2，并将模型参数、随机种子和 provenance 独立保存。禁止将缺失字段静默填 0 或常数后标称来自 Alibaba。
""",
    )
    heterogeneity.to_csv(output / "05_gpu_heterogeneity_audit.csv", index=False)
    write_text(
        output / "06_gpu_heterogeneity_adapter_design.md",
        f"""# GPU 异构性适配设计

## 事实

- GPU2026 官方说明峰值为 155,410 张 GPU、37,707 台 GPU server，但本轮未下载 351.8 GB pod archive，因此没有伪造分型号计数。[官方 README]({GPU2026_README})
- SpotGPU2026 本地完整文件包含 {len(nodes):,} 个节点、{nodes['gpu_model'].nunique()} 类 GPU；型号与任务请求均直接存在。[官方 README]({SPOT2026_README})
- 当前 SustainCluster 只有 `total_gpus/gpu_req` 数量约束，不支持任务型号兼容性或分型号 DC 容量。

## 方案 A：第一阶段兼容模式

保留 `gpu_type` 为 provenance/evaluation metadata，但调度资源仍使用 `gpu_request` 的 GPU-equivalent 数量。该模式只用于验证 arrival、duration 与 CPU/GPU request 分布；报告必须写明忽略型号兼容性，不能据此评价异构放置质量。

## 方案 B：异构 GPU 扩展模式

后续在独立版本中加入 `task.required_gpu_type`、`DC.gpu_type_capacity` 与 feasible mask 兼容性检查。该扩展会改变 simulator/MPC 约束，不属于本轮。

推荐：v3 第一版若启动，先采用方案 A 做数据链验证，同时保留原始型号；异构调度必须另立实验版本。
""",
    )
    priority.to_csv(output / "07_priority_task_type_audit.csv", index=False)
    write_text(
        output / "08_defer_semantics_feasibility.md",
        f"""# Defer 语义可行性

GPU2026 将 HP 定义为 guaranteed resources，将 LP 定义为使用 flexible capacity 的 low-priority/spot；还区分 training、online/offline inference 与 dev。[官方 schema]({GPU2026_SCHEMA})。SpotGPU2026 将 HP 描述为 strict-SLO workload，将 Spot 描述为使用 opportunistic spot instances 的任务。[官方 README]({SPOT2026_README})。

## 可以支持的谨慎映射

- `online_inference` 与 HP：默认不可任意 defer；需要严格 SLA。
- `offline_inference`、部分 training/dev、LP/Spot：可以作为“较高 delay tolerance”的候选类。
- Spot 表明资源获取方式与优先级，不等于任务可无限等待，也不直接给出 deadline、最大延迟或抢占代价。

## 尚需建模

数值 SLA、deadline、最大 defer 次数、抢占/恢复代价、不同类型的 waiting penalty 均未由 trace 直接给出。本轮不修改 reward 或 waiting penalty。旧场景 teacher defer=0% 不能仅靠把 `Spot` 映成 `deferrable=True` 自动修复。
""",
    )
    time_table.to_csv(output / "09_time_resolution_audit.csv", index=False)
    write_text(
        output / "10_time_resolution_recommendation.md",
        "# 时间尺度建议\n\n"
        "- Alibaba2020：继续使用当前已验证的 15 分钟主线。\n"
        "- GPU2026：使用 1 小时外部场景验证。`day+hour` 只表示 pod 在该小时被观测到，不是到达时间；禁止拆成四个所谓真实 15 分钟事件。若研究小时内 arrival，必须命名 `SYNTHETIC WITHIN-HOUR ARRIVAL MODEL`。\n"
        "- SpotGPU2026：保留相对秒级 `submit_time` 为原始证据；当前 simulator 可用确定性的 floor-to-15-minute 分箱，但必须保存原始秒与 bin offset。\n\n"
        + markdown_table(time_table),
    )
    write_text(
        output / "11_download_license_audit.md",
        f"""# 下载与许可审计

审计日期：{AUDIT_DATE}；官方仓库 master 固定到 `{UPSTREAM_COMMIT}`。

## 下载

- GPU2026 官方提供四个 ZIP：pod 351,803,513,445 B、server 3,080,121,387 B、network 203,904,100 B、execution summary 1,188,295,031 B。[下载清单]({GPU2026_DOWNLOAD})
- 官方文档列出可直接访问的公开下载入口，未说明需要额外申请、账号或访问令牌；这只确认获取方式，不构成再分发或商业使用授权。
- 本轮未下载 GPU2026 fact tables；原因是适配审计不需要盲目获取约 356 GB 数据，且仓库没有小型 fact sample。
- SpotGPU2026 两个 CSV 由官方 GitHub 下载：job 23,055,929 B，node 78,330 B；完整 SHA 见 `12_raw_data_manifest.csv`。
- 原始文件位于 `.gitignore` 已覆盖的 `data/raw/alibaba_2026/`，未修改源文件。

## 许可

- [项目根 README]({ROOT_README})明确鼓励将 traces 用于 study/research，两个 2026 README 都要求引用论文。
- GitHub repository API 的 `license` 为 `null`，仓库根目录 `LICENSE` 返回 404。
- 研究使用意图：**CONFIRMED BY README**。
- 原始数据重新分发：**NOT CONFIRMED**。
- 转换后数据随项目公开：**NOT CONFIRMED**。
- 商业使用、再许可、派生数据权利：**NOT CONFIRMED**。

结论：可以在私有研究环境继续技术验证，但公开发布原始/转换数据前必须获得明确许可说明；代码与 schema 报告可与数据文件分离发布。
""",
    )
    raw.to_csv(output / "12_raw_data_manifest.csv", index=False)
    gpu_sample.to_csv(output / "13_gpu2026_canonical_sample.csv", index=False)
    spot_sample.to_csv(output / "14_spotgpu2026_canonical_sample.csv", index=False)
    provenance.to_csv(output / "15_canonical_field_provenance.csv", index=False)
    compatibility.to_csv(output / "16_sustaincluster_compatibility_matrix.csv", index=False)
    write_text(
        output / "17_mpc_expert_v3_schema_proposal.md",
        "# MPC Expert Dataset v3 Schema Proposal\n\n"
        "## 推荐场景\n\n"
        "- Scenario A: `Alibaba2020 repaired baseline`，保持与 v2 可比。\n"
        "- Scenario B: `SpotGPU2026 task-level validation`，保留真实 relative-second arrival、duration、CPU/GPU、worker 与 GPU type；memory、bandwidth、origin、SLA 和 estimated duration 必须使用显式版本化模型。\n"
        "- Scenario C: `GPU2026 hourly external validation`，只验证 workload mix、priority、GPU heterogeneity、resource distribution 与 topology，不作为真实 15 分钟任务流。\n\n"
        "## 信息隔离\n\n"
        "State/task 当前可部署字段进入主表；`true_duration` 仅供 simulator；H1/Oracle action 仅作 label；所有 future arrivals/price/carbon 必须写入独立 `privileged_future/`，不得与 deployable observation 混列。\n\n"
        "## 启动前锁定项\n\n"
        "1. Spot CPU/GPU request 的 per-worker/per-job scope。\n"
        "2. host-memory 与 bandwidth 模型及其版本。\n"
        "3. estimated-duration 的 train-only 拟合和 episode split。\n"
        "4. HP/Spot 到 SLA/defer 的保守映射。\n"
        "5. 数据公开与派生数据许可边界。\n",
    )
    schema.to_csv(output / "18_mpc_expert_v3_schema.csv", index=False)
    write_text(
        output / "19_recommended_data_strategy.md",
        """# 推荐数据策略

## THREE-SOURCE DATA STRATEGY

- Alibaba2020：继续承担机制连续性、15 分钟主线和历史结果可比性。
- SpotGPU2026：作为最有价值的现代任务级验证源；arrival/duration/CPU/GPU/worker/GPU model 真实保留，缺失字段单独建模。
- GPU2026：作为超大规模小时级外部验证源；用于 workload mix、priority、GPU heterogeneity、resource fragmentation、topology 与网络分布，不冒充任务到达 trace。

这不是把三套数据拼成同一“真实 trace”。每个 scenario 都必须保留 `source_dataset`、原始时间语义、DIRECT/DERIVED/MODELED provenance 和独立结论边界。

当前不直接重建 v3：先完成 Spot resource-scope、memory/bandwidth、duration estimator、SLA/defer 与许可五项合同冻结，再进入正式生成。
""",
    )
    span_days = (jobs["submit_time"].max() - jobs["submit_time"].min()) / 86_400
    write_text(
        output / "20_final_diagnosis.md",
        f"""# 最终诊断

## GPU2026

**SUITABLE FOR HOURLY / SCENARIO VALIDATION ONLY**

原因：185 天 hourly pod/server 表没有任务级精确 arrival；execution summary 可经 pod_id 提供观测 duration；CPU/GPU/priority/task type/GPU model/topology 很强，但 host-memory request 缺失，network 仅有 day 109..115 的 server-hour aggregate。本轮只有官方文档证据，没有本地 fact-table 统计。

## SpotGPU2026

**SUITABLE WITH LIMITED FIELD MODELING**

原因：本地完整文件验证 {len(jobs):,} jobs、{len(nodes):,} nodes、{nodes['gpu_model'].nunique()} GPU models、{span_days:.6f} 天相对秒级时间轴；arrival、duration、CPU/GPU、worker、GPU model 与 HP/Spot 均可用。host memory、bandwidth、task type、origin DC、数值 SLA 与 estimated duration 缺失，CPU/GPU request scope 还需官方确认。

## 数据覆盖证据

| 数据源 | 时间覆盖 | jobs/pods | nodes/servers | GPU models | priority classes | task types | 证据级别 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPU2026 | day 0..184（185 天） | 未本地统计 | 37,707（官方小时峰值） | 有公开桶，未本地计数 | 3 | 6 | DOCUMENTATION_LEVEL_ONLY |
| SpotGPU2026 | {span_days:.6f} 天 | {len(jobs):,} | {len(nodes):,} | {nodes['gpu_model'].nunique()} | {jobs['job_type'].nunique()} | MISSING | LOCAL_FULL_FILE |

## 总体

**THREE-SOURCE DATA STRATEGY**

技术上值得继续为 v3 做准备，但当前 `Ready to rebuild v3 = NO`。最小 canonical adapter 已证明 Spot 记录可以无损保留直接字段，同时会因缺少当前必需 memory/bandwidth 等字段而明确停在 `MISSING`，这正是下一次合同决策的边界。
""",
    )
    write_text(
        output / "21_summary.md",
        f"""# Alibaba 2026 GPU 数据适配可行性审计 v1

1. **GPU2026 能恢复任务级到达吗？** 不能。`day+hour` 是 pod-hour 观测，不是 arrival timestamp。
2. **SpotGPU2026 能恢复任务级到达吗？** 能。`submit_time` 是相对首任务提交的秒数；本地文件单调、无缺失，跨度 {span_days:.6f} 天。
3. **哪套最适合当前 task-level adapter？** SpotGPU2026，但只能在缺失字段模型显式化后接入。
4. **duration 能作 true_duration 吗？** GPU2026 `duration_hours` 与 Spot `duration` 都可作为观测 ground truth 候选；需质量过滤，只供 simulator，不能给在线策略。
5. **estimated_duration 怎么处理？** 使用仅在 train episodes 拟合的历史 estimator；可先做全局/类型条件中位数基线，再单独研究 predictor。
6. **memory/bandwidth 缺什么？** 两套 2026 都缺 task host-memory request；Spot 无 network/bandwidth；GPU2026 只有 GPU memory 与 server-hour rx/tx，均不能替代当前字段。
7. **GPU 异构如何映射？** v3 首阶段用 GPU-equivalent 兼容模式并保留型号 metadata；异构兼容约束另立 simulator/MPC 版本。
8. **priority/type 能支持 SLA/defer 吗？** 能指导类别设计，不能直接给数值 deadline 或无限 defer；这些仍是场景模型。
9. **15 分钟适合吗？** Alibaba2020 适合；GPU2026 不适合直接 15 分钟；Spot 可由真实秒级事件确定性分箱。
10. **GPU2026 的定位？** 小时级外部验证，不是当前 task-level arrival 输入。
11. **v3 应含哪些 scenario？** A=2020 repaired baseline，B=Spot task-level validation，C=GPU2026 hourly external validation。
12. **现在值得正式重建 v3 吗？** 值得准备，但尚未 ready；先冻结 request scope、memory/bandwidth、duration estimator、SLA/defer 和许可合同。
""",
    )


def copy_raw_samples(output: Path, jobs: pd.DataFrame, nodes: pd.DataFrame) -> None:
    destination = output / "raw_samples"
    destination.mkdir(parents=True, exist_ok=True)
    jobs.head(20).to_csv(destination / "spotgpu2026_job_head20.csv", index=False)
    nodes.head(20).to_csv(destination / "spotgpu2026_node_head20.csv", index=False)
    for name in (
        "gpu2026_README.md",
        "gpu2026_schema.md",
        "gpu2026_data_download.md",
        "spotgpu2026_README.md",
    ):
        shutil.copyfile(DOC_ROOT / name, destination / name)
    write_text(
        destination / "gpu2026_fact_sample_status.md",
        "# GPU2026 Fact Sample Status\n\nNo fact row was downloaded. The official code repository contains no small fact-table sample, and the pod archive is 351,803,513,445 bytes. `13_gpu2026_canonical_sample.csv` is therefore a DOCUMENTATION_LEVEL_ONLY schema template with MISSING values, not a fabricated record.",
    )


def build_manifest(
    output: Path,
    workspace: dict[str, Any],
    jobs: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    metadata = json.loads((DOC_ROOT / "repository_metadata.json").read_text(encoding="utf-8"))
    commit = json.loads((DOC_ROOT / "master_commit.json").read_text(encoding="utf-8"))
    if metadata.get("license") is not None or commit.get("sha") != UPSTREAM_COMMIT:
        raise RuntimeError("upstream repository metadata changed from audited snapshot")
    manifest = {
        "audit": "Alibaba 2026 GPU 数据适配可行性审计 v1",
        "audit_date": AUDIT_DATE,
        "workspace": workspace,
        "upstream": {
            "repository": "https://github.com/alibaba/clusterdata",
            "commit": UPSTREAM_COMMIT,
            "github_license_metadata": None,
            "root_license_file": "NOT FOUND",
        },
        "alibaba2020_frozen_baseline": {
            "local_path": str(ALIBABA_2020.relative_to(ROOT)).replace("\\", "/"),
            "file_size_bytes": ALIBABA_2020_SIZE,
            "sha256": ALIBABA_2020_SHA256,
            "shape": [37_552, 2],
            "columns": ["interval_15m", "tasks_matrix"],
        },
        "spotgpu2026_local_statistics": {
            "jobs": len(jobs),
            "nodes": len(nodes),
            "gpu_models": int(nodes["gpu_model"].nunique()),
            "priority_counts": {key: int(value) for key, value in jobs["job_type"].value_counts().items()},
            "submit_time_monotonic": bool(jobs["submit_time"].is_monotonic_increasing),
            "submit_time_min_seconds": float(jobs["submit_time"].min()),
            "submit_time_max_seconds": float(jobs["submit_time"].max()),
            "duration_min_seconds": float(jobs["duration"].min()),
            "duration_max_seconds": float(jobs["duration"].max()),
            "missing_values": int(jobs.isna().sum().sum() + nodes.isna().sum().sum()),
        },
        "gpu2026_evidence_level": "DOCUMENTATION_LEVEL_ONLY",
        "gpu2026_fact_tables_downloaded": False,
        "gpu2026_verdict": "SUITABLE FOR HOURLY / SCENARIO VALIDATION ONLY",
        "spotgpu2026_verdict": "SUITABLE WITH LIMITED FIELD MODELING",
        "recommended_strategy": "THREE-SOURCE DATA STRATEGY",
        "ready_to_rebuild_v3": False,
        "expert_dataset_v3_generated": False,
        "mpc_run": False,
        "transformer_training": False,
        "bc_training": False,
        "rl_sac": False,
    }
    manifest["artifact_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "audit_manifest.json"
    }
    write_json(output / "audit_manifest.json", manifest)


def run(output: Path = DEFAULT_OUTPUT) -> Path:
    workspace = validate_frozen_workspace()
    jobs, nodes = validate_sources()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "raw_samples").mkdir(exist_ok=True)

    target = target_schema()
    mapping = source_mapping()
    heterogeneity = gpu_heterogeneity(jobs, nodes)
    priority = priority_task_type(jobs)
    time_table = time_resolution()
    gpu_sample = gpu2026_canonical_template()
    spot_sample = spot_canonical_sample(jobs)
    provenance = canonical_provenance()
    compatibility = compatibility_matrix()
    schema = v3_schema()
    raw = raw_manifest()

    write_reports(
        output,
        jobs,
        nodes,
        target,
        mapping,
        heterogeneity,
        priority,
        time_table,
        gpu_sample,
        spot_sample,
        provenance,
        compatibility,
        schema,
        raw,
    )
    copy_raw_samples(output, jobs, nodes)
    plot_outputs(output, jobs, heterogeneity)
    build_manifest(output, workspace, jobs, nodes)
    print("gpu2026=SUITABLE FOR HOURLY / SCENARIO VALIDATION ONLY")
    print("spotgpu2026=SUITABLE WITH LIMITED FIELD MODELING")
    print("strategy=THREE-SOURCE DATA STRATEGY")
    print("ready_to_rebuild_v3=NO")
    print(f"artifacts={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
