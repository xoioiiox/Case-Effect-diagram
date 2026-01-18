# 依赖库安装（首次运行前执行）
# pip install notears pgmpy pandas numpy matplotlib networkx scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from notears-learn import notears_mlp
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer

# -------------------------- 1. 数据准备：生成干净的混合变量数据集 --------------------------
# 模拟目标场景变量：包含离散变量（装备类型、是否开火）和连续变量（探测精度、跟踪稳定性、命中概率）
# 真实因果关系：装备类型→探测精度→跟踪稳定性→命中概率；是否开火→命中概率
np.random.seed(42)  # 固定随机种子，确保结果可复现
n_samples = 1000  # 样本量

# 1.1 生成离散变量
equipment_type = np.random.binomial(1, 0.5, n_samples)  # 装备类型：0=老装备，1=新装备（50%概率）
is_fire = np.random.binomial(1, 0.6, n_samples)        # 是否开火：0=否，1=是（60%概率）

# 1.2 生成连续变量（基于真实因果关系）
detection_accuracy = 0.3 * equipment_type + np.random.normal(0.5, 0.1, n_samples)  # 新装备探测精度更高
tracking_stability = 0.4 * detection_accuracy + np.random.normal(0.4, 0.08, n_samples)  # 探测精度影响跟踪稳定性
hit_prob = 0.5 * tracking_stability + 0.2 * is_fire + np.random.normal(0.3, 0.05, n_samples)  # 跟踪稳定性+开火影响命中概率

# 1.3 整理为DataFrame（列顺序与变量名对应）
data = pd.DataFrame({
    "equipment_type": equipment_type,
    "detection_accuracy": detection_accuracy,
    "tracking_stability": tracking_stability,
    "is_fire": is_fire,
    "hit_prob": hit_prob
})

# 打印数据基本信息（验证数据生成正常）
print("="*50)
print("数据集基本信息：")
print(data.head())
print(f"\n数据集形状：{data.shape}")
print(f"变量类型：")
print(f"- 离散变量：equipment_type（装备类型）、is_fire（是否开火）")
print(f"- 连续变量：detection_accuracy（探测精度）、tracking_stability（跟踪稳定性）、hit_prob（命中概率）")
print("="*50)

# -------------------------- 2. 物理黑名单定义：过滤违禁边 --------------------------
# 定义违背物理/逻辑规则的违禁边（如因果倒置、无关联边）
variables = data.columns.tolist()  # 变量名称列表（与数据列顺序一致）
n_vars = len(variables)

# 2.1 黑名单规则：key为“源节点→目标节点”，value=1表示禁止
blacklist_rules = {
    "hit_prob→detection_accuracy": 1,    # 禁止：命中概率→探测精度（因果倒置）
    "is_fire→equipment_type": 1,         # 禁止：是否开火→装备类型（无逻辑关联）
    "tracking_stability→equipment_type": 1,  # 禁止：跟踪稳定性→装备类型（无逻辑关联）
    "hit_prob→equipment_type": 1         # 禁止：命中概率→装备类型（无逻辑关联）
}

# 2.2 转换为黑名单矩阵（N×N，1=禁止，0=允许），适配NOTEARS-MLP输入
blacklist_matrix = np.zeros((n_vars, n_vars))
for forbidden_edge, flag in blacklist_rules.items():
    if flag == 1:
        src_name, tgt_name = forbidden_edge.split("→")
        src_idx = variables.index(src_name)
        tgt_idx = variables.index(tgt_name)
        blacklist_matrix[src_idx, tgt_idx] = 1  # 标记该方向为禁止边

print("\n物理黑名单矩阵（1=禁止，0=允许）：")
print(pd.DataFrame(blacklist_matrix, index=variables, columns=variables))
print("="*50)

# -------------------------- 3. NOTEARS-MLP引擎：挖掘因果结构 --------------------------
# 核心功能：基于混合变量数据，挖掘因果边，同时遵守黑名单约束（禁止违禁边）
print("\n开始运行NOTEARS-MLP因果挖掘...")

# 3.1 数据预处理：提取数据矩阵（shape：n_samples × n_vars）
X = data.values

# 3.2 配置NOTEARS-MLP参数
var_types = [1, 0, 0, 1, 0]  # 变量类型：1=离散，0=连续（对应data列顺序）
lambda1 = 0.1  # L1正则化参数（控制边的稀疏性，避免过多冗余边）
alpha = 0.01   # 无环约束权重（确保输出是有向无环图DAG）

# 3.3 训练NOTEARS-MLP模型（传入黑名单矩阵，禁止违禁边）
W = notears_mlp(
    X,
    var_types=var_types,
    lambda1=lambda1,
    alpha=alpha,
    blacklist=blacklist_matrix,  # 应用物理黑名单约束
    max_iter=1000,  # 最大迭代次数
    h_tol=1e-8      # 无环约束容忍度
)

# 3.4 提取有效因果边（过滤权重极小的边，阈值设为1e-5）
causal_edges = []
for i in range(n_vars):
    for j in range(n_vars):
        if W[i, j] > 1e-5:  # 仅保留权重显著的边
            src_var = variables[i]
            tgt_var = variables[j]
            causal_edges.append((src_var, tgt_var, round(W[i, j], 3)))  # 存储（源变量，目标变量，权重）

# 打印挖掘结果
print("NOTEARS-MLP挖掘的因果边（含权重）：")
for edge in causal_edges:
    print(f"  {edge[0]} → {edge[1]} （权重：{edge[2]}）")
print("="*50)

# -------------------------- 4. 贝叶斯网络构建：生成带CPT的概率模型 --------------------------
# 核心功能：基于NOTEARS-MLP的因果结构，构建贝叶斯网络，计算条件概率表（CPT）
print("\n开始构建贝叶斯网络...")

# 4.1 定义贝叶斯网络结构（仅保留NOTEARS-MLP挖掘的因果边）
bn_model = BayesianNetwork()
bn_model.add_nodes_from(variables)  # 添加所有变量节点
bn_edges = [(src, tgt) for src, tgt, weight in causal_edges]  # 提取因果边（不含权重）
bn_model.add_edges_from(bn_edges)  # 添加因果边

# 4.2 离散化连续变量（贝叶斯网络CPT需基于离散变量计算）
# 对连续变量（探测精度、跟踪稳定性、命中概率）进行3分箱离散化（低、中、高）
data_discrete = data.copy()
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')  # 分位数分箱
continuous_cols = ["detection_accuracy", "tracking_stability", "hit_prob"]
data_discrete[continuous_cols] = discretizer.fit_transform(data[continuous_cols])
data_discrete[continuous_cols] = data_discrete[continuous_cols].astype(int)  # 转换为整数标签（0=低，1=中，2=高）

# 4.3 学习条件概率表（CPT）：使用最大似然估计
bn_model.fit(
    data_discrete,
    estimator=MaximumLikelihoodEstimator  # 最大似然估计（适合干净数据）
)

# 打印贝叶斯网络信息
print("贝叶斯网络结构：")
print(f"节点：{bn_model.nodes()}")
print(f"边：{bn_model.edges()}")
print("\n关键节点的条件概率表（CPT）：")
# 打印“命中概率”的CPT（核心目标变量）
cpt_hit = bn_model.get_cpds("hit_prob")
print("命中概率（hit_prob）的CPT：")
print(cpt_hit)
print("="*50)

# -------------------------- 5. 结果可视化：因果图与贝叶斯网络结构 --------------------------
# 5.1 绘制NOTEARS-MLP挖掘的因果图
plt.figure(figsize=(10, 6))
G = nx.DiGraph()
G.add_nodes_from(variables)
G.add_edges_from([(src, tgt) for src, tgt, weight in causal_edges])
# 布局：spring_layout（基于力导向，使图结构清晰）
pos = nx.spring_layout(G, seed=42, k=2)  # k控制节点间距
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", edgecolors="black", linewidths=1.5)
# 绘制边
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="darkred", linewidths=1.5)
# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")
# 添加标题和网格
plt.title("NOTEARS-MLP挖掘的因果图（含物理黑名单约束）", fontsize=14, pad=20)
plt.axis("off")  # 隐藏坐标轴
plt.tight_layout()
plt.savefig("causal_graph.png", dpi=300, bbox_inches="tight")  # 保存图片
plt.show()

# 5.2 绘制贝叶斯网络结构（与因果图一致，突出概率模型属性）
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(bn_model, seed=42, k=2)
# 绘制节点（区分离散/连续变量）
discrete_nodes = ["equipment_type", "is_fire"]
continuous_nodes = ["detection_accuracy", "tracking_stability", "hit_prob"]
nx.draw_networkx_nodes(bn_model, pos, nodelist=discrete_nodes, node_size=3000, node_color="lightcoral", edgecolors="black", linewidths=1.5)
nx.draw_networkx_nodes(bn_model, pos, nodelist=continuous_nodes, node_size=3000, node_color="lightgreen", edgecolors="black", linewidths=1.5)
# 绘制边
nx.draw_networkx_edges(bn_model, pos, arrowstyle="->", arrowsize=20, edge_color="darkblue", linewidths=1.5)
# 绘制标签
nx.draw_networkx_labels(bn_model, pos, font_size=11, font_weight="bold")
# 添加图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='离散变量'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='连续变量')
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
plt.title("贝叶斯网络结构（含离散/连续变量区分）", fontsize=14, pad=20)
plt.axis("off")
plt.tight_layout()
plt.savefig("bayesian_network.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------- 6. 简单因果推理：验证贝叶斯网络实用性 --------------------------
print("\n贝叶斯网络因果推理示例：")
infer = VariableElimination(bn_model)
# 推理：新装备（equipment_type=1）+ 开火（is_fire=1）时，命中概率的分布
query_result = infer.query(
    variables=["hit_prob"],
    evidence={"equipment_type": 1, "is_fire": 1}
)
print("新装备 + 开火 时，命中概率分布（0=低，1=中，2=高）：")
print(query_result)

# 推理：老装备（equipment_type=0）+ 不开火（is_fire=0）时，命中概率的分布
query_result2 = infer.query(
    variables=["hit_prob"],
    evidence={"equipment_type": 0, "is_fire": 0}
)
print("\n老装备 + 不开火 时，命中概率分布（0=低，1=中，2=高）：")
print(query_result2)
print("="*50)

# -------------------------- 运行说明 --------------------------
print("\nDemo运行完成！生成以下文件：")
print("1. causal_graph.png：NOTEARS-MLP挖掘的因果图（含物理黑名单约束）")
print("2. bayesian_network.png：贝叶斯网络结构可视化图")
print("\n核心验证点：")
print("- 因果图应包含预设的真实因果边，无黑名单中的违禁边")
print("- 贝叶斯网络CPT应合理（如“新装备+开火”时，命中概率为“高”的概率更高）")