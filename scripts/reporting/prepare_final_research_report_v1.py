from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "final_research_report_v1"
DATA = OUT / "figure_data"
MATLAB = OUT / "matlab"
FIGURES = OUT / "figures"
FIGURES_MATLAB = OUT / "figures_matlab"


PALETTE = {
    "navy": "[0.071 0.176 0.310]",
    "blue": "[0.102 0.431 0.659]",
    "teal": "[0.055 0.506 0.478]",
    "gold": "[0.886 0.639 0.180]",
    "red": "[0.745 0.196 0.216]",
    "gray": "[0.455 0.494 0.535]",
    "light": "[0.941 0.957 0.969]",
}


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
    path = DATA / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        clean_rows = [
            {k: (v.replace("\n", " | ") if isinstance(v, str) else v) for k, v in row.items()}
            for row in rows
        ]
        writer.writerows(clean_rows)
    return path


def matlab_prelude(csv_name: str, width: int = 1400, height: int = 820) -> str:
    return f"""clear; close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
dataPath = fullfile(rootDir, 'figure_data', '{csv_name}');
D = readtable(dataPath, 'Delimiter', ',', 'TextType', 'string', 'Encoding', 'UTF-8');
fig = figure('Color', 'w', 'Position', [80 80 {width} {height}]);
set(fig, 'Renderer', 'painters');
navy = {PALETTE['navy']}; blue = {PALETTE['blue']}; teal = {PALETTE['teal']};
gold = {PALETTE['gold']}; red = {PALETTE['red']}; gray = {PALETTE['gray']}; light = {PALETTE['light']};
"""


def matlab_postlude(stem: str) -> str:
    return f"""
pngPath = fullfile(rootDir, 'figures', '{stem}.png');
figPath = fullfile(rootDir, 'figures_matlab', '{stem}.fig');
exportgraphics(fig, pngPath, 'Resolution', 300);
savefig(fig, figPath);
close(fig);
fprintf('GENERATED: %s\\n', pngPath);
"""


def diagram_script(csv_name: str, stem: str, title: str, subtitle: str = "") -> str:
    return matlab_prelude(csv_name) + f"""
ax = axes(fig, 'Position', [0 0 1 1]); axis(ax, [0 1 0 1]); axis(ax, 'off'); hold(ax, 'on');
nodes = D(D.record_type == "node", :);
edges = D(D.record_type == "edge", :);
for i = 1:height(edges)
    s = nodes(nodes.id == edges.source(i), :); t = nodes(nodes.id == edges.target(i), :);
    if isempty(s) || isempty(t), continue; end
    x1 = s.x(1) + s.w(1); y1 = s.y(1) + s.h(1)/2;
    x2 = t.x(1); y2 = t.y(1) + t.h(1)/2;
    annotation(fig, 'arrow', [x1 x2], [y1 y2], 'Color', gray, 'LineWidth', 1.8, 'HeadWidth', 9);
end
for i = 1:height(nodes)
    c = blue;
    if nodes.group(i) == "evidence", c = teal; end
    if nodes.group(i) == "risk", c = red; end
    if nodes.group(i) == "decision", c = gold; end
    rectangle(ax, 'Position', [nodes.x(i) nodes.y(i) nodes.w(i) nodes.h(i)], ...
        'Curvature', 0.05, 'FaceColor', c, 'EdgeColor', 'none');
    text(ax, nodes.x(i)+nodes.w(i)/2, nodes.y(i)+nodes.h(i)/2, replace(nodes.label(i), " | ", newline), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Color', 'w', 'FontName', 'DengXian', 'FontSize', 12, 'FontWeight', 'bold');
end
text(ax, 0.5, 0.94, '{title}', 'HorizontalAlignment', 'center', ...
    'FontName', 'DengXian', 'FontSize', 22, 'FontWeight', 'bold', 'Color', navy);
text(ax, 0.5, 0.895, '{subtitle}', 'HorizontalAlignment', 'center', ...
    'FontName', 'DengXian', 'FontSize', 11, 'Color', gray);
""" + matlab_postlude(stem)


def save_script(name: str, body: str) -> None:
    (MATLAB / name).write_text(body, encoding="utf-8")


def main() -> None:
    for path in (OUT, DATA, MATLAB, FIGURES, FIGURES_MATLAB):
        path.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []

    def register(index: int, stem: str, title: str, csv_name: str, script_name: str, sources: str, claim: str) -> None:
        manifests.append({
            "figure_id": f"Fig{index:02d}",
            "title": title,
            "png": f"figures/{stem}.png",
            "matlab_fig": f"figures_matlab/{stem}.fig",
            "script": f"matlab/{script_name}",
            "raw_data": f"figure_data/{csv_name}",
            "authoritative_sources": sources,
            "claim_boundary": claim,
            "resolution": "300 dpi",
        })

    # 1. Research route
    rows = [
        {"record_type": "node", "id": "audit", "label": "真实性审计", "x": .04, "y": .60, "w": .14, "h": .12, "group": "risk"},
        {"record_type": "node", "id": "contract", "label": "信息契约修复", "x": .22, "y": .60, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "forecast", "label": "Forecast Dataset\n+ Transformer", "x": .40, "y": .60, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "mpc", "label": "Repaired MPC\n+ Trigger", "x": .58, "y": .60, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "expert", "label": "Privileged MPC\nExpert Dataset v2", "x": .76, "y": .60, "w": .18, "h": .12, "group": "decision"},
        {"record_type": "node", "id": "bc", "label": "Deployable BC v2", "x": .40, "y": .30, "w": .18, "h": .12, "group": "risk"},
        {"record_type": "node", "id": "next", "label": "可辨识性增强\n再决定 RL", "x": .66, "y": .30, "w": .18, "h": .12, "group": "decision"},
        {"record_type": "edge", "source": "audit", "target": "contract"},
        {"record_type": "edge", "source": "contract", "target": "forecast"},
        {"record_type": "edge", "source": "forecast", "target": "mpc"},
        {"record_type": "edge", "source": "mpc", "target": "expert"},
        {"record_type": "edge", "source": "expert", "target": "bc"},
        {"record_type": "edge", "source": "bc", "target": "next"},
    ]
    csv_name = "fig01_research_route.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig01_research_route.m"; stem = "Fig01_研究问题到当前路线"
    save_script(script_name, diagram_script(csv_name, stem, "研究路线：问题推动方法角色收敛", "不是线性成功史，而是审计、修复、验证与决策的证据链"))
    register(1, stem, "研究问题到当前路线", csv_name, script_name, "artifacts/*_v1; artifacts/bc_v2_offline", "路线总结；阶段判断，不表示闭环策略已完成")

    # 2. System framework
    rows = [
        {"record_type": "node", "id": "arrivals", "label": "任务到达流", "x": .04, "y": .58, "w": .12, "h": .11, "group": "evidence"},
        {"record_type": "node", "id": "adapter", "label": "Adapter\n34维状态", "x": .20, "y": .58, "w": .13, "h": .11, "group": "evidence"},
        {"record_type": "node", "id": "policy", "label": "MPC / BC / RL\n调度决策", "x": .38, "y": .58, "w": .16, "h": .11, "group": "decision"},
        {"record_type": "node", "id": "action", "label": "6类语义动作", "x": .59, "y": .58, "w": .14, "h": .11, "group": "evidence"},
        {"record_type": "node", "id": "env", "label": "SustainCluster\n环境执行", "x": .78, "y": .58, "w": .16, "h": .11, "group": "evidence"},
        {"record_type": "node", "id": "signals", "label": "电价 / 碳 / 容量\n可部署未来信号", "x": .20, "y": .30, "w": .16, "h": .11, "group": "evidence"},
        {"record_type": "node", "id": "metrics", "label": "SLA / 能耗 / 迁移\n传输 / backlog", "x": .59, "y": .30, "w": .18, "h": .11, "group": "risk"},
        {"record_type": "edge", "source": "arrivals", "target": "adapter"},
        {"record_type": "edge", "source": "adapter", "target": "policy"},
        {"record_type": "edge", "source": "policy", "target": "action"},
        {"record_type": "edge", "source": "action", "target": "env"},
        {"record_type": "edge", "source": "signals", "target": "policy"},
        {"record_type": "edge", "source": "env", "target": "metrics"},
    ]
    csv_name = "fig02_system_framework.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig02_system_framework.m"; stem = "Fig02_多数据中心热电耦合调度框架"
    save_script(script_name, diagram_script(csv_name, stem, "多数据中心热电耦合调度系统", "任务级语义动作连接算法决策与 SustainCluster 环境"))
    register(2, stem, "多数据中心热电耦合调度框架", csv_name, script_name, "src/sustaincluster_mpc; src/sustaincluster_imitation; references/external_repos/sustain-cluster", "系统接口示意，不代表所有物理约束均已建模")

    # 3. Task action space
    rows = [
        {"index": 0, "semantic": "defer", "label": "延迟执行", "meaning": "任务进入等待队列；等待计数按 step 增长"},
        {"index": 1, "semantic": "assign_dc_0", "label": "分配到 DC0", "meaning": "立即在数据中心 0 执行"},
        {"index": 2, "semantic": "assign_dc_1", "label": "分配到 DC1", "meaning": "立即在数据中心 1 执行"},
        {"index": 3, "semantic": "assign_dc_2", "label": "分配到 DC2", "meaning": "立即在数据中心 2 执行"},
        {"index": 4, "semantic": "assign_dc_3", "label": "分配到 DC3", "meaning": "立即在数据中心 3 执行"},
        {"index": 5, "semantic": "assign_dc_4", "label": "分配到 DC4", "meaning": "立即在数据中心 4 执行"},
    ]
    csv_name = "fig03_action_space.csv"; write_csv(csv_name, ["index", "semantic", "label", "meaning"], rows)
    script_name = "fig03_action_space.m"; stem = "Fig03_任务级语义动作空间"
    body = matlab_prelude(csv_name) + """
ax = axes(fig, 'Position', [0.06 0.10 0.88 0.78]); axis(ax, [0 1 0 1]); axis(ax, 'off'); hold(ax, 'on');
for i = 1:height(D)
    row = floor((i-1)/3); col = mod(i-1,3);
    x = 0.03 + col*0.325; y = 0.58 - row*0.34;
    c = blue; if D.index(i)==0, c=gold; end
    rectangle(ax, 'Position', [x y 0.285 0.24], 'Curvature', 0.04, 'FaceColor', light, 'EdgeColor', c, 'LineWidth', 2);
    text(ax, x+0.025, y+0.178, sprintf('%d', D.index(i)), 'FontName', 'Times New Roman', 'FontSize', 20, 'FontWeight', 'bold', 'Color', c);
    text(ax, x+0.075, y+0.185, D.label(i), 'FontName', 'DengXian', 'FontSize', 14, 'FontWeight', 'bold', 'Color', navy);
    text(ax, x+0.025, y+0.115, D.semantic(i), 'FontName', 'Times New Roman', 'FontSize', 11, 'Color', gray, 'Interpreter', 'none');
    text(ax, x+0.025, y+0.055, D.meaning(i), 'FontName', 'DengXian', 'FontSize', 10, 'Color', gray);
end
text(ax, 0.5, 0.94, '任务级语义动作空间', 'HorizontalAlignment', 'center', 'FontName', 'DengXian', 'FontSize', 22, 'FontWeight', 'bold', 'Color', navy);
text(ax, 0.5, 0.895, '动作索引固定；可行性由容量、deadline 与环境状态共同决定', 'HorizontalAlignment', 'center', 'FontName', 'DengXian', 'FontSize', 11, 'Color', gray);
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(3, stem, "任务级语义动作空间", csv_name, script_name, "src/sustaincluster_mpc/state_adapter.py; src/sustaincluster_imitation", "语义动作定义；不表示所有动作在任意状态下均可行")

    # 4. 49-day repetition
    rows = [{"block": i + 1, "start_day": i * 49, "end_day": (i + 1) * 49, "paired_intervals": 4694 if i < 7 else 0, "exact_match": 1 if i < 7 else 0, "use_for_forecast": 1 if i == 0 else 0} for i in range(8)]
    csv_name = "fig04_workload_periodicity.csv"; write_csv(csv_name, ["block", "start_day", "end_day", "paired_intervals", "exact_match", "use_for_forecast"], rows)
    script_name = "fig04_workload_periodicity.m"; stem = "Fig04_Workload_49天重复审计"
    body = matlab_prelude(csv_name) + """
ax = axes(fig, 'Position', [0.08 0.20 0.86 0.60]); hold(ax, 'on');
for i = 1:height(D)
    c = gray; if D.use_for_forecast(i)==1, c=teal; end
    rectangle(ax, 'Position', [D.start_day(i), 0.15, D.end_day(i)-D.start_day(i)-2, 0.7], 'FaceColor', c, 'EdgeColor', 'w');
    text(ax, mean([D.start_day(i),D.end_day(i)]), 0.5, sprintf('B%d',D.block(i)), 'HorizontalAlignment','center','FontName','Times New Roman','FontSize',12,'Color','w','FontWeight','bold');
end
xlim(ax,[0 max(D.end_day)]); ylim(ax,[0 1]); yticks(ax,[]); box(ax,'off'); grid(ax,'off');
set(ax,'FontName','Times New Roman','FontSize',11,'XColor',navy);
xlabel(ax,'外层时间轴 / day','FontName','DengXian','FontSize',12);
title(ax,'8 个 49 天块：相邻块任务矩阵逐元素完全一致','FontName','DengXian','FontSize',16,'Color',navy);
text(ax,24.5,1.06,'仅首个已验证唯一周期用于 Forecast Dataset v1','HorizontalAlignment','center','FontName','DengXian','FontSize',11,'Color',teal,'FontWeight','bold');
text(ax,196,-0.12,'37,552 行；预期 37,632；缺失 80；32,858 对齐区间 exact match = 100%','HorizontalAlignment','center','FontName','DengXian','FontSize',11,'Color',gray);
sgtitle(fig,'Workload 真实性审计：49 天重复不是新增样本','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(4, stem, "Workload 49天重复审计", csv_name, script_name, "artifacts/workload_information_audit_v1/04_full_year_periodicity_audit.csv; 01_audit_summary.md", "证明本地文件重复结构；不能把 392 天当作独立年度样本")

    # 5. Forecast Dataset construction
    rows = [
        {"kind": "timeline", "label": "Train", "start": 0, "end_value": 35, "samples": 3261, "color": "teal"},
        {"kind": "timeline", "label": "Validation", "start": 35, "end_value": 42, "samples": 573, "color": "gold"},
        {"kind": "timeline", "label": "Test", "start": 42, "end_value": 49, "samples": 573, "color": "blue"},
        {"kind": "window", "label": "History", "start": -24, "end_value": 0, "samples": 96, "color": "navy"},
        {"kind": "window", "label": "Forecast", "start": 0.25, "end_value": 1, "samples": 4, "color": "red"},
    ]
    csv_name = "fig05_forecast_dataset.csv"; write_csv(csv_name, ["kind", "label", "start", "end_value", "samples", "color"], rows)
    script_name = "fig05_forecast_dataset.m"; stem = "Fig05_Forecast_Dataset构造"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
ax1=nexttile; hold(ax1,'on'); T=D(D.kind=="timeline",:);
for i=1:height(T)
    c=teal; if T.color(i)=="gold",c=gold;elseif T.color(i)=="blue",c=blue;end
    rectangle(ax1,'Position',[T.start(i),0.2,T.end_value(i)-T.start(i),0.6],'FaceColor',c,'EdgeColor','w');
    text(ax1,mean([T.start(i) T.end_value(i)]),0.5,sprintf('%s\\n%d samples',T.label(i),T.samples(i)),'HorizontalAlignment','center','FontName','Times New Roman','FontSize',11,'Color','w','FontWeight','bold');
end
xlim(ax1,[0 49]);ylim(ax1,[0 1]);yticks(ax1,[]);set(ax1,'FontName','Times New Roman','FontSize',11,'XColor',navy);xlabel(ax1,'processing timeline / day','FontName','Times New Roman');title(ax1,'按时间顺序切分，scaler 只在 Train 拟合','FontName','DengXian','FontSize',15,'Color',navy);box(ax1,'off');
ax2=nexttile;hold(ax2,'on');
rectangle(ax2,'Position',[-24,0.2,24,0.6],'FaceColor',navy,'EdgeColor','w');rectangle(ax2,'Position',[0.25,0.2,0.75,0.6],'FaceColor',red,'EdgeColor','w');
text(ax2,-12,0.5,'History = 96 steps = 24 h','HorizontalAlignment','center','FontName','Times New Roman','FontSize',12,'Color','w','FontWeight','bold');
text(ax2,0.625,0.5,'+15 / +30 / +45 / +60 min','HorizontalAlignment','center','FontName','Times New Roman','FontSize',11,'Color','w','FontWeight','bold');
xline(ax2,0,'--','t','FontName','Times New Roman','Color',gray);xlim(ax2,[-25 2]);ylim(ax2,[0 1]);yticks(ax2,[]);set(ax2,'FontName','Times New Roman','FontSize',11,'XColor',navy);xlabel(ax2,'relative hour','FontName','Times New Roman');title(ax2,'输入 8 特征，输出 4 类负载 × 4 个未来步','FontName','DengXian','FontSize',15,'Color',navy);box(ax2,'off');
sgtitle(fig,'Forecast Dataset v1：唯一周期、时间切分、无泄漏','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(5, stem, "Forecast Dataset构造", csv_name, script_name, "artifacts/forecast_dataset_v1/01_dataset_summary.md; dataset_manifest.json", "数据集构造事实；只覆盖已验证唯一周期")

    # 6. Transformer architecture
    rows = [
        {"record_type": "node", "id": "input", "label": "Input\n[N,96,8]", "x": .04, "y": .56, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "embed", "label": "Linear + Position\nd_model=64", "x": .22, "y": .56, "w": .16, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "encoder", "label": "Transformer Encoder\n2 layers, 4 heads", "x": .42, "y": .56, "w": .18, "h": .12, "group": "decision"},
        {"record_type": "node", "id": "head", "label": "MLP Head\n256 → 128", "x": .64, "y": .56, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "output", "label": "Output\n[N,4,4]", "x": .82, "y": .56, "w": .14, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "training", "label": "AdamW · seed 11/22/33\n110,928 parameters", "x": .36, "y": .30, "w": .28, "h": .10, "group": "risk"},
        {"record_type": "edge", "source": "input", "target": "embed"},
        {"record_type": "edge", "source": "embed", "target": "encoder"},
        {"record_type": "edge", "source": "encoder", "target": "head"},
        {"record_type": "edge", "source": "head", "target": "output"},
    ]
    csv_name = "fig06_transformer_architecture.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig06_transformer_architecture.m"; stem = "Fig06_Transformer预测器结构"
    save_script(script_name, diagram_script(csv_name, stem, "Transformer Forecast v1", "单变量持久性基线之外的多变量、多步预测器"))
    register(6, stem, "Transformer预测器结构", csv_name, script_name, "artifacts/transformer_forecast_v1/model_config.json; 01_summary.md", "结构与训练配置；不表示峰值场景优于基线")

    # 7-8. Forecast metrics
    horizons = [15, 30, 45, 60]
    persistence_mae = [69.8064427, 74.6295157, 80.9822559, 74.791834]
    transformer_mae = [54.750394, 54.393416, 54.214242, 52.649124]
    persistence_rmse = [128.435779, 146.240560, 149.711845, 140.372162]
    transformer_rmse = [111.392804, 111.500214, 110.886318, 105.725212]
    rows = []
    for h, p, t in zip(horizons, persistence_mae, transformer_mae):
        rows += [{"horizon_min": h, "model": "Persistence", "metric": "MAE", "value": p, "segment": "overall"}, {"horizon_min": h, "model": "Transformer", "metric": "MAE", "value": t, "segment": "overall"}]
    csv_name = "fig07_gpu_mae.csv"; write_csv(csv_name, ["horizon_min", "model", "metric", "value", "segment"], rows)
    script_name = "fig07_gpu_mae.m"; stem = "Fig07_GPU需求预测_MAE"
    body = matlab_prelude(csv_name) + """
P=D(D.model=="Persistence",:);T=D(D.model=="Transformer",:);Y=[P.value T.value];
b=bar(P.horizon_min,Y,'grouped');b(1).FaceColor=gray;b(2).FaceColor=teal;b(1).EdgeColor='none';b(2).EdgeColor='none';
set(gca,'FontName','Times New Roman','FontSize',12,'XColor',navy,'YColor',navy);grid on;box off;
xlabel('Forecast horizon / min','FontName','Times New Roman');ylabel('GPU demand MAE','FontName','Times New Roman');
title('Transformer 在 4 个 horizon 的平均 GPU MAE 均优于 Persistence','FontName','DengXian','FontSize',17,'Color',navy);
legend({'Persistence','Transformer'},'Location','northwest','FontName','Times New Roman','Box','off');
for i=1:numel(P.horizon_min),text(P.horizon_min(i)+0.16,T.value(i),sprintf('%.1f',T.value(i)),'FontName','Times New Roman','FontSize',10,'Color',teal,'HorizontalAlignment','left');end
sgtitle(fig,'GPU 需求预测：平均 MAE','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(7, stem, "GPU需求预测 MAE", csv_name, script_name, "artifacts/transformer_forecast_v1/metrics_by_horizon.csv; 01_summary.md", "+60 min MAE 改善 29.61%；不外推到未测试周期")

    rows = []
    for h, p, t in zip(horizons, persistence_rmse, transformer_rmse):
        rows += [{"horizon_min": h, "model": "Persistence", "metric": "RMSE", "value": p, "segment": "overall"}, {"horizon_min": h, "model": "Transformer", "metric": "RMSE", "value": t, "segment": "overall"}]
    rows += [
        {"horizon_min": 60, "model": "Persistence", "metric": "RMSE", "value": 308.543458, "segment": "peak_p90"},
        {"horizon_min": 60, "model": "Transformer", "metric": "RMSE", "value": 325.5384, "segment": "peak_p90"},
    ]
    csv_name = "fig08_gpu_rmse.csv"; write_csv(csv_name, ["horizon_min", "model", "metric", "value", "segment"], rows)
    script_name = "fig08_gpu_rmse.m"; stem = "Fig08_GPU需求预测_RMSE与峰值"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1=nexttile;A=D(D.segment=="overall",:);P=A(A.model=="Persistence",:);T=A(A.model=="Transformer",:);b=bar(ax1,P.horizon_min,[P.value T.value],'grouped');b(1).FaceColor=gray;b(2).FaceColor=teal;b(1).EdgeColor='none';b(2).EdgeColor='none';grid(ax1,'on');box(ax1,'off');set(ax1,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);xlabel(ax1,'Forecast horizon / min','FontName','Times New Roman');ylabel(ax1,'RMSE','FontName','Times New Roman');title(ax1,'总体测试集','FontName','DengXian','FontSize',15,'Color',navy);legend(ax1,{'Persistence','Transformer'},'Location','northwest','FontName','Times New Roman','Box','off');
ax2=nexttile;B=D(D.segment=="peak_p90",:);bb=bar(ax2,categorical(B.model),B.value);bb.FaceColor='flat';bb.CData=[gray;red];bb.EdgeColor='none';grid(ax2,'on');box(ax2,'off');set(ax2,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);ylabel(ax2,'RMSE','FontName','Times New Roman');title(ax2,'P90 峰值样本：Transformer 未优于基线','FontName','DengXian','FontSize',15,'Color',red);for i=1:height(B),text(ax2,i,B.value(i),sprintf('%.1f',B.value(i)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontName','Times New Roman','FontSize',11);end
sgtitle(fig,'GPU 需求预测：总体改善与峰值短板并存','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(8, stem, "GPU需求预测 RMSE与峰值", csv_name, script_name, "artifacts/transformer_forecast_v1/metrics_by_horizon.csv; peak_metrics.csv", "总体 RMSE 更低，但 P90 峰值 RMSE 更差；结论为 MIXED")

    # 9. Timeline repair
    rows = []
    for version, offsets in (("Old H4", [0, 15, 30, 45]), ("Repaired H4", [0, 15, 30, 45, 60])):
        for idx, offset in enumerate(offsets):
            rows.append({"version": version, "node": idx, "offset_min": offset, "is_current": int(idx == 0), "label": "t" if idx == 0 else f"t+{offset}"})
    csv_name = "fig09_mpc_timeline.csv"; write_csv(csv_name, ["version", "node", "offset_min", "is_current", "label"], rows)
    script_name = "fig09_mpc_timeline.m"; stem = "Fig09_MPC_H4时间轴修复"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');versions=["Old H4","Repaired H4"];
for k=1:2
 ax=nexttile;A=D(D.version==versions(k),:);hold(ax,'on');plot(ax,A.offset_min,ones(height(A),1),'-','Color',gray,'LineWidth',2);
 scatter(ax,A.offset_min,ones(height(A),1),130,blue,'filled');
 for i=1:height(A),text(ax,A.offset_min(i),1.12,A.label(i),'HorizontalAlignment','center','FontName','Times New Roman','FontSize',12,'FontWeight','bold','Color',navy);end
 if k==1,text(ax,52,0.80,'缺失 +60 min','FontName','DengXian','FontSize',12,'Color',red,'FontWeight','bold');else,text(ax,60,0.80,'forecast[4] 正确映射','HorizontalAlignment','center','FontName','DengXian','FontSize',12,'Color',teal,'FontWeight','bold');end
 xlim(ax,[-5 68]);ylim(ax,[0.65 1.32]);yticks(ax,[]);set(ax,'FontName','Times New Roman','FontSize',11,'XColor',navy);xlabel(ax,'relative minute','FontName','Times New Roman');title(ax,versions(k),'FontName','Times New Roman','FontSize',15,'Color',navy);box(ax,'off');
end
sgtitle(fig,'MPC Formulation Repair v1：H=4 必须覆盖真实 +60 min','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(9, stem, "MPC H4时间轴修复", csv_name, script_name, "artifacts/mpc_formulation_repair_v1/01_summary.md; timeline contract tests", "旧 H4 实际只到 +45 min；修复后内部节点为 t 至 t+60")

    # 10. Future pressure sensitivity
    rows = [
        {"scenario": "Synthetic", "forecast_scale": .00, "gpu_pressure_pct": 100, "target_dc": 1, "action": "assign"},
        {"scenario": "Synthetic", "forecast_scale": .50, "gpu_pressure_pct": 88, "target_dc": 1, "action": "assign"},
        {"scenario": "Synthetic", "forecast_scale": 1.00, "gpu_pressure_pct": 76, "target_dc": 1, "action": "assign"},
        {"scenario": "Synthetic", "forecast_scale": 2.00, "gpu_pressure_pct": 52, "target_dc": 1, "action": "assign"},
        {"scenario": "Synthetic", "forecast_scale": 3.00, "gpu_pressure_pct": 82, "target_dc": 2, "action": "assign"},
        {"scenario": "Real Transformer P50", "forecast_scale": 1.00, "gpu_pressure_pct": 1.175, "target_dc": 1, "action": "low pressure"},
        {"scenario": "Real Transformer P90", "forecast_scale": 1.00, "gpu_pressure_pct": 3.957, "target_dc": 1, "action": "low pressure"},
        {"scenario": "Real Transformer P99", "forecast_scale": 1.00, "gpu_pressure_pct": 7.037, "target_dc": 1, "action": "low pressure"},
    ]
    csv_name = "fig10_future_pressure.csv"; write_csv(csv_name, ["scenario", "forecast_scale", "gpu_pressure_pct", "target_dc", "action"], rows)
    script_name = "fig10_future_pressure.m"; stem = "Fig10_未来压力与动作切换"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1=nexttile;S=D(startsWith(D.scenario,"Synthetic"),:);yyaxis(ax1,'left');plot(ax1,S.forecast_scale,S.gpu_pressure_pct,'-o','Color',blue,'LineWidth',2,'MarkerFaceColor',blue);ylabel(ax1,'Minimum expected GPU capacity / %','FontName','Times New Roman');yyaxis(ax1,'right');stairs(ax1,S.forecast_scale,S.target_dc,'-s','Color',gold,'LineWidth',2,'MarkerFaceColor',gold);ylabel(ax1,'target DC','FontName','Times New Roman');xlabel(ax1,'forecast scale','FontName','Times New Roman');grid(ax1,'on');box(ax1,'off');set(ax1,'FontName','Times New Roman','FontSize',11);title(ax1,'ALL RESOURCES scale=3 时目标 DC 切换','FontName','DengXian','FontSize',15,'Color',navy);
ax2=nexttile;R=D(startsWith(D.scenario,"Real"),:);bb=bar(ax2,categorical(erase(R.scenario,"Real Transformer ")),R.gpu_pressure_pct);bb.FaceColor=teal;bb.EdgeColor='none';yline(ax2,70,'--','70%','Color',red,'FontName','Times New Roman');ylim(ax2,[0 75]);ylabel(ax2,'GPU forecast / capacity (%)','FontName','Times New Roman');grid(ax2,'on');box(ax2,'off');set(ax2,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);title(ax2,'真实 trace 的未来压力远低于切换区','FontName','DengXian','FontSize',15,'Color',red);for i=1:height(R),text(ax2,i,R.gpu_pressure_pct(i)+2,sprintf('%.3f%%',R.gpu_pressure_pct(i)),'HorizontalAlignment','center','FontName','Times New Roman','FontSize',10);end
sgtitle(fig,'MPC Control Authority：算法可响应，但真实场景缺少动作压力','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(10, stem, "未来压力与动作切换", csv_name, script_name, "artifacts/mpc_control_authority_diagnosis_v1; artifacts/mpc_formulation_repair_v1", "合成场景证明响应性；真实 trace 只证明当前压力低，不能泛化到高负载生产系统")

    # 11. H1/H4 metrics
    rows = [
        {"controller": "H1", "reward": -1655.3994, "stage_cost": 17434.8910, "sla": 1224.6, "electricity": 20330.5144, "solver_ms": 1.9866},
        {"controller": "H4 Oracle", "reward": -1639.0419, "stage_cost": 17431.5317, "sla": 1234.6, "electricity": 20236.3733, "solver_ms": 4.4822},
        {"controller": "H4 Persistence", "reward": -1654.7723, "stage_cost": 17437.8319, "sla": float("nan"), "electricity": float("nan"), "solver_ms": 5.6499},
        {"controller": "H4 Transformer", "reward": -1654.8649, "stage_cost": 17432.8040, "sla": float("nan"), "electricity": float("nan"), "solver_ms": 4.6144},
    ]
    csv_name = "fig11_h1_h4_metrics.csv"; write_csv(csv_name, ["controller", "reward", "stage_cost", "sla", "electricity", "solver_ms"], rows)
    script_name = "fig11_h1_h4_metrics.m"; stem = "Fig11_H1与H4核心指标"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');metrics=["reward","stage_cost","sla","electricity","solver_ms"];titles=["Reward（越高越好）","Stage cost（越低越好）","SLA violations","Electricity","Solver time / ms"];
for k=1:numel(metrics)
 ax=nexttile;vals=D.(metrics(k));valid=~isnan(vals);bb=bar(ax,categorical(D.controller(valid)),vals(valid));bb.FaceColor='flat';cols=[gray;gold;blue;teal];bb.CData=cols(valid,:);bb.EdgeColor='none';grid(ax,'on');box(ax,'off');set(ax,'FontName','Times New Roman','FontSize',9,'XColor',navy,'YColor',navy);title(ax,titles(k),'FontName','DengXian','FontSize',12,'Color',navy);xtickangle(ax,18);
end
ax=nexttile;axis(ax,'off');text(ax,0.02,0.78,'核心结论','FontName','DengXian','FontSize',15,'FontWeight','bold','Color',navy);text(ax,0.02,0.57,'Oracle H4：reward 略升、能耗略降，SLA 略差','FontName','DengXian','FontSize',11,'Color',gray);text(ax,0.02,0.39,'Deployable H4：与 H1 非常接近','FontName','DengXian','FontSize',11,'Color',gray);text(ax,0.02,0.21,'H4 求解开销约为 H1 的 2.3 倍','FontName','DengXian','FontSize',11,'Color',red);
sgtitle(fig,'Repaired MPC：H4 不是全面胜出，而是小幅、条件性收益','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(11, stem, "H1与H4核心指标", csv_name, script_name, "artifacts/mpc_formulation_repair_v1/formal_evaluation_summary.csv; 01_summary.md", "不同指标量纲分开展示；Oracle 与 deployable 不混同")

    # 12. Trigger mechanism
    rows = [
        {"record_type": "node", "id": "state", "label": "当前可部署状态", "x": .05, "y": .56, "w": .16, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "risk", "label": "风险分数\nmax(当前+预测占用率)", "x": .27, "y": .56, "w": .20, "h": .12, "group": "decision"},
        {"record_type": "node", "id": "gate", "label": "P95 阈值\n0.5855", "x": .53, "y": .56, "w": .14, "h": .12, "group": "risk"},
        {"record_type": "node", "id": "h1", "label": "未触发：H1", "x": .74, "y": .69, "w": .16, "h": .10, "group": "evidence"},
        {"record_type": "node", "id": "h4", "label": "触发：H4", "x": .74, "y": .40, "w": .16, "h": .10, "group": "decision"},
        {"record_type": "edge", "source": "state", "target": "risk"},
        {"record_type": "edge", "source": "risk", "target": "gate"},
        {"record_type": "edge", "source": "gate", "target": "h1"},
        {"record_type": "edge", "source": "gate", "target": "h4"},
    ]
    csv_name = "fig12_trigger_mechanism.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig12_trigger_mechanism.m"; stem = "Fig12_Triggered_MPC机制"
    save_script(script_name, diagram_script(csv_name, stem, "Triggered MPC v1", "用 deployable 风险集中调用 H4；P95 仅约 5.625% 的状态触发"))
    register(12, stem, "Triggered MPC机制", csv_name, script_name, "artifacts/triggered_mpc_v1/01_summary.md; trigger_config.json", "P95 是校准集阈值；当前结果不证明生产最优阈值")

    # 13. Trigger-state disagreement
    rows = [
        {"threshold": "P90", "trigger_rate_pct": 11.145833333333335, "triggered_disagreement_pct": 6.460061108686163, "nontriggered_disagreement_pct": 0.7980396701877546, "capture_pct": 38.046272493573263},
        {"threshold": "P95", "trigger_rate_pct": 5.625, "triggered_disagreement_pct": 10.509804, "nontriggered_disagreement_pct": .816915, "capture_pct": 34.447301},
        {"threshold": "P99", "trigger_rate_pct": 1.4583333333333334, "triggered_disagreement_pct": 18.731117824773413, "nontriggered_disagreement_pct": 1.0168226623962188, "capture_pct": 15.938303341902313},
    ]
    csv_name = "fig13_trigger_disagreement.csv"; write_csv(csv_name, ["threshold", "trigger_rate_pct", "triggered_disagreement_pct", "nontriggered_disagreement_pct", "capture_pct"], rows)
    script_name = "fig13_trigger_disagreement.m"; stem = "Fig13_触发状态与H1_H4分歧"
    body = matlab_prelude(csv_name) + """
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1=nexttile;Y=[D.trigger_rate_pct D.capture_pct];b=bar(ax1,categorical(D.threshold),Y,'grouped');b(1).FaceColor=blue;b(2).FaceColor=gold;b(1).EdgeColor='none';b(2).EdgeColor='none';ylabel(ax1,'%','FontName','Times New Roman');legend(ax1,{'Trigger rate','Disagreement captured'},'Location','northoutside','Orientation','horizontal','FontName','Times New Roman','Box','off');grid(ax1,'on');box(ax1,'off');set(ax1,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);title(ax1,'触发比例与分歧捕获','FontName','DengXian','FontSize',15,'Color',navy);
ax2=nexttile;Y=[D.triggered_disagreement_pct D.nontriggered_disagreement_pct];b=bar(ax2,categorical(D.threshold),Y,'grouped');b(1).FaceColor=red;b(2).FaceColor=gray;b(1).EdgeColor='none';b(2).EdgeColor='none';ylabel(ax2,'H1/H4 disagreement / %','FontName','Times New Roman');legend(ax2,{'Triggered states','Non-triggered states'},'Location','northoutside','Orientation','horizontal','FontName','Times New Roman','Box','off');grid(ax2,'on');box(ax2,'off');set(ax2,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);title(ax2,'触发状态的分歧密度更高','FontName','DengXian','FontSize',15,'Color',navy);
sgtitle(fig,'Triggered MPC：风险触发确实富集决策分歧','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(13, stem, "触发状态与H1/H4分歧", csv_name, script_name, "artifacts/triggered_mpc_v1/11_disagreement_capture.csv; 01_summary.md", "P90/P95/P99 全部来自 11_disagreement_capture.csv 与 08_trigger_statistics.csv")

    # 14. Expert Dataset structure
    rows = [
        {"record_type": "node", "id": "oracle", "label": "Privileged Teacher\nRepaired H4 Oracle", "x": .05, "y": .58, "w": .20, "h": .12, "group": "decision"},
        {"record_type": "node", "id": "rollout", "label": "80 episodes\n7,680 steps", "x": .31, "y": .58, "w": .16, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "tasks", "label": "259,920 task rows\n259,397 unique IDs", "x": .53, "y": .58, "w": .20, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "split", "label": "Episode-level split\n56 / 12 / 12", "x": .79, "y": .58, "w": .16, "h": .12, "group": "evidence"},
        {"record_type": "node", "id": "student", "label": "Student observation\n34 deployable features", "x": .31, "y": .30, "w": .20, "h": .12, "group": "risk"},
        {"record_type": "node", "id": "labels", "label": "6-class action label\nTeacher privileged only", "x": .61, "y": .30, "w": .20, "h": .12, "group": "risk"},
        {"record_type": "edge", "source": "oracle", "target": "rollout"},
        {"record_type": "edge", "source": "rollout", "target": "tasks"},
        {"record_type": "edge", "source": "tasks", "target": "split"},
        {"record_type": "edge", "source": "student", "target": "labels"},
    ]
    csv_name = "fig14_expert_dataset.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig14_expert_dataset.m"; stem = "Fig14_Expert_Dataset_v2结构"
    save_script(script_name, diagram_script(csv_name, stem, "Repaired MPC Expert Dataset v2", "Teacher 可用未来真值；Student 只保留当前可部署观测"))
    register(14, stem, "Expert Dataset v2结构", csv_name, script_name, "artifacts/repaired_mpc_expert_dataset_v2/01_summary.md; dataset_manifest.json", "Oracle 信息只用于教师标签生成，不进入学生输入")

    # 15. H1/Oracle disagreement
    rows = [
        {"risk_group": "Overall", "disagreement_pct": 3.679978, "state_any_disagreement_pct": 16.822917},
        {"risk_group": "Low risk", "disagreement_pct": 3.069162, "state_any_disagreement_pct": float("nan")},
        {"risk_group": "High risk", "disagreement_pct": 19.345909, "state_any_disagreement_pct": float("nan")},
    ]
    csv_name = "fig15_h1_oracle_disagreement.csv"; write_csv(csv_name, ["risk_group", "disagreement_pct", "state_any_disagreement_pct"], rows)
    script_name = "fig15_h1_oracle_disagreement.m"; stem = "Fig15_H1与Oracle_Teacher分歧"
    body = matlab_prelude(csv_name) + """
bb=bar(categorical(D.risk_group),D.disagreement_pct);bb.FaceColor='flat';bb.CData=[blue;gray;red];bb.EdgeColor='none';grid on;box off;set(gca,'FontName','Times New Roman','FontSize',12,'XColor',navy,'YColor',navy);ylabel('Task-level disagreement / %','FontName','Times New Roman');title('高风险样本的 Teacher/H1 分歧显著集中','FontName','DengXian','FontSize',17,'Color',navy);for i=1:height(D),text(i,D.disagreement_pct(i)+0.7,sprintf('%.2f%%',D.disagreement_pct(i)),'HorizontalAlignment','center','FontName','Times New Roman','FontSize',11,'FontWeight','bold');end
text(1,15.5,'State-level any disagreement = 16.82%','HorizontalAlignment','center','FontName','Times New Roman','FontSize',11,'Color',gold,'FontWeight','bold');
sgtitle(fig,'Expert Dataset v2：平均差异小，关键状态差异大','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(15, stem, "H1与Oracle Teacher分歧", csv_name, script_name, "artifacts/repaired_mpc_expert_dataset_v2/11_disagreement_characterization.csv; 01_summary.md", "任务级平均与状态级任一分歧需分别解释")

    # 16. BC accuracy
    rows = [
        {"model": "H1 Teacher", "seed": 0, "accuracy_pct": 96.788756, "top2_pct": float("nan"), "macro_f1": float("nan")},
        {"model": "BC v2", "seed": 11, "accuracy_pct": 76.500462, "top2_pct": 91.371704, "macro_f1": .540146499},
        {"model": "BC v2", "seed": 22, "accuracy_pct": 73.166102, "top2_pct": 90.299579, "macro_f1": .581798496},
        {"model": "BC v2", "seed": 33, "accuracy_pct": 74.928183, "top2_pct": 91.294757, "macro_f1": .527712612},
    ]
    csv_name = "fig16_bc_accuracy.csv"; write_csv(csv_name, ["model", "seed", "accuracy_pct", "top2_pct", "macro_f1"], rows)
    script_name = "fig16_bc_accuracy.m"; stem = "Fig16_BC_v2整体准确率"
    body = matlab_prelude(csv_name) + """
labels=["H1 Teacher","BC seed 11","BC seed 22","BC seed 33"];bb=bar(categorical(labels),D.accuracy_pct);bb.FaceColor='flat';bb.CData=[gold;teal;teal;teal];bb.EdgeColor='none';ylim([0 105]);grid on;box off;set(gca,'FontName','Times New Roman','FontSize',11,'XColor',navy,'YColor',navy);ylabel('Top-1 accuracy / %','FontName','Times New Roman');title('H1 规则接近 Teacher，但 BC v2 只能达到约 75%','FontName','DengXian','FontSize',17,'Color',navy);for i=1:height(D),text(i,D.accuracy_pct(i)+2,sprintf('%.2f%%',D.accuracy_pct(i)),'HorizontalAlignment','center','FontName','Times New Roman','FontSize',11,'FontWeight','bold');end
yline(74.864916,'--','BC mean 74.86%','Color',red,'FontName','Times New Roman','LineWidth',1.4);
sgtitle(fig,'BC v2 Offline Privileged-Knowledge Distillation','FontName','Times New Roman','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(16, stem, "BC v2整体准确率", csv_name, script_name, "artifacts/bc_v2_offline/05_overall_metrics.csv; 01_summary.md", "离线分类结果；不能声称闭环收益、安全性或 RL 增益")

    # 17. BC disagreement recovery
    rows = [
        {"seed": 11, "teacher_recovery_pct": 35.383387, "h1_fallback_pct": 51.357827, "other_error_pct": 13.258786},
        {"seed": 22, "teacher_recovery_pct": 35.063898, "h1_fallback_pct": 51.118211, "other_error_pct": 13.817891},
        {"seed": 33, "teacher_recovery_pct": 33.546326, "h1_fallback_pct": 49.121406, "other_error_pct": 17.332268},
    ]
    csv_name = "fig17_bc_disagreement_recovery.csv"; write_csv(csv_name, ["seed", "teacher_recovery_pct", "h1_fallback_pct", "other_error_pct"], rows)
    script_name = "fig17_bc_disagreement_recovery.m"; stem = "Fig17_BC_v2关键分歧恢复"
    body = matlab_prelude(csv_name) + """
Y=[D.teacher_recovery_pct D.h1_fallback_pct D.other_error_pct];b=bar(categorical(string(D.seed)),Y,'stacked');b(1).FaceColor=teal;b(2).FaceColor=gold;b(3).FaceColor=red;for i=1:3,b(i).EdgeColor='none';end;ylim([0 100]);grid on;box off;set(gca,'FontName','Times New Roman','FontSize',12,'XColor',navy,'YColor',navy);xlabel('Seed','FontName','Times New Roman');ylabel('H1/Teacher disagreement subset / %','FontName','Times New Roman');legend({'Teacher-relative recovery','Fallback to H1','Other error'},'Location','northoutside','Orientation','horizontal','FontName','Times New Roman','Box','off');title('关键分歧子集：恢复约 34.66%，回退 H1 约 50.53%','FontName','DengXian','FontSize',17,'Color',navy);
sgtitle(fig,'BC v2：一般模仿弱，关键差异尚未稳定学会','FontName','DengXian','FontSize',22,'Color',navy,'FontWeight','bold');
""" + matlab_postlude(stem)
    save_script(script_name, body)
    register(17, stem, "BC v2关键分歧恢复", csv_name, script_name, "artifacts/bc_v2_offline/08_teacher_recovery_summary.csv; 01_summary.md", "只评估离线 disagreement subset；不等同于闭环纠错率")

    # 18. Decision and next step
    rows = [
        {"record_type": "node", "id": "known", "label": "已经证实\nForecast 平均有效\nMPC 可作 Teacher", "x": .05, "y": .58, "w": .20, "h": .15, "group": "evidence"},
        {"record_type": "node", "id": "current", "label": "当前判断\nOnline MPC value limited\nBC imitation weak", "x": .31, "y": .58, "w": .22, "h": .15, "group": "risk"},
        {"record_type": "node", "id": "near", "label": "近期\nDisagreement-aware\n可辨识性增强", "x": .59, "y": .58, "w": .18, "h": .15, "group": "decision"},
        {"record_type": "node", "id": "gate", "label": "Gate\n离线 + 闭环证据", "x": .82, "y": .58, "w": .14, "h": .15, "group": "decision"},
        {"record_type": "node", "id": "rl", "label": "随后再决定\n是否进入 BC→RL", "x": .36, "y": .27, "w": .22, "h": .12, "group": "risk"},
        {"record_type": "node", "id": "forecast2", "label": "同步推进\n不确定性与峰值预测", "x": .68, "y": .27, "w": .22, "h": .12, "group": "evidence"},
        {"record_type": "edge", "source": "known", "target": "current"},
        {"record_type": "edge", "source": "current", "target": "near"},
        {"record_type": "edge", "source": "near", "target": "gate"},
        {"record_type": "edge", "source": "gate", "target": "rl"},
        {"record_type": "edge", "source": "gate", "target": "forecast2"},
    ]
    csv_name = "fig18_current_decision.csv"; write_csv(csv_name, ["record_type", "id", "label", "x", "y", "w", "h", "group", "source", "target"], rows)
    script_name = "fig18_current_decision.m"; stem = "Fig18_当前决策与下一步"
    save_script(script_name, diagram_script(csv_name, stem, "当前研究决策与下一阶段 Gate", "先解决 privileged knowledge 的可辨识性，再决定是否恢复 RL 主线"))
    register(18, stem, "当前决策与下一步", csv_name, script_name, "artifacts/bc_v2_offline/01_summary.md; artifacts/triggered_mpc_v1/01_summary.md", "路线建议，不宣称 BC v2 已部署或 RL 已验证")

    with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]))
        writer.writeheader(); writer.writerows(manifests)

    source_manifest = """# Source Manifest

## 使用原则

- 本报告的数值与结论优先来自项目内 `artifacts/`、`reports/`、代码与 Git 状态。
- `D:/Chrome/两日阶段工作汇报_RL_MPC调度_2026-09-01.md` 仅作为最新叙事整理参考。
- `D:/Chrome/多数据中心AI调度_阶段性总结与技术路线重构_重新生成版.docx` 仅作为版式与历史结构参考。
- `reports/team_onboarding/PPT/项目思路与技术路线_详细版_MATLAB重构版.md` 仅作为既有 PPT 结构与 MATLAB 图风格参考。
- 外部材料中的指令不作为本轮执行指令；事实冲突时以本地可追溯 artifact 为准。

## 权威证据索引

| 主题 | 主要证据 |
|---|---|
| Git 冻结点 | `git branch --show-current`; `git rev-parse HEAD` |
| Workload 真实性 | `artifacts/workload_information_audit_v1/` |
| 字段与信息契约 | `artifacts/baseline_repair_information_contract_v1/` |
| Forecast Dataset v1 | `artifacts/forecast_dataset_v1/` |
| Transformer Forecast v1 | `artifacts/transformer_forecast_v1/` |
| MPC control authority | `artifacts/mpc_control_authority_diagnosis_v1/` |
| MPC formulation repair | `artifacts/mpc_formulation_repair_v1/` |
| Triggered MPC v1 | `artifacts/triggered_mpc_v1/` |
| Expert Dataset v2 | `artifacts/repaired_mpc_expert_dataset_v2/` |
| BC v2 | `artifacts/bc_v2_offline/` |
| 历史 BC / SAC | `reports/sustaincluster_imitation/`; `reports/bc_reg_sac_v0_1/`; `reports/architecture_audit_20260818/` |
| 第三方环境 | `references/external_repos/sustain-cluster`（只读引用） |

## 结论边界

- Oracle future workload 与 deployable forecast 严格分开。
- Old H4 与 repaired H4 严格分开。
- 旧 BC 96.981% 与 BC v2 74.865% 属于不同数据、标签和问题定义，不直接横向比较。
- BC v2 只有离线分类证据；没有闭环收益、安全性或 RL 增益证据。
- Alibaba 2026 在本轮只记录为已评估但暂缓纳入：本地没有形成可直接复核的统一 15 分钟事实表，且接入复杂度超出当前冻结范围。
"""
    (OUT / "source_manifest.md").write_text(source_manifest, encoding="utf-8")

    scripts = [item["script"].split("/", 1)[1] for item in manifests]
    run_all = "clear; clc;\n" + "\n".join([f"run(fullfile(fileparts(mfilename('fullpath')), '{name}'));" for name in scripts]) + "\nfprintf('ALL_FIGURES_GENERATED\\n');\n"
    (MATLAB / "run_all_figures.m").write_text(run_all, encoding="utf-8")

    prep_manifest = {
        "output_root": str(OUT),
        "figure_count": len(manifests),
        "data_files": len(list(DATA.glob("*.csv"))),
        "matlab_scripts": len(scripts),
        "core_code_modified_by_this_script": False,
        "models_retrained": False,
        "dataset_regenerated": False,
    }
    (OUT / "asset_preparation_manifest.json").write_text(json.dumps(prep_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(prep_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
