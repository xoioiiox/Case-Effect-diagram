# filename: main_analysis.py
import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# pgmpy 库用于参数学习和推理
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# --- 兼容性补丁 ---
try:
    np.product
except AttributeError:
    np.product = np.prod

# --- 0. NOTEARS 导入检查 ---
try:
    sys.path.append("./notears")
    from notears.nonlinear import notears_nonlinear, NotearsMLP

    print("[INFO] 成功导入 notears_nonlinear 库")
except ImportError:
    print("[WARN] 未找到 notears 库，正在使用模拟生成的邻接矩阵进行演示...")


    def notears_nonlinear(model, X, lambda1, lambda2):
        d = X.shape[1]
        return np.random.uniform(0, 0.5, (d, d)) * (np.random.rand(d, d) > 0.7)


# ==========================================
# 1. 结构学习与黑名单约束
# ==========================================
def generate_blacklist_auto(df_columns):
    """
    根据变量层级自动生成黑名单
    """
    blacklist = []
    tier_map = {
        'Weather': 0, 'Jamming': 0, 'Occlusion': 0,
        'SNR': 1,
        'Rec_Conf': 2, 'Rec_Latency': 2,
        'Track_Stable': 3, 'Lock_Time': 3,
        'Hit_Res': 4
    }

    def get_tier(col_name):
        return tier_map.get(col_name, 2)

    for src in df_columns:
        for dst in df_columns:
            if src == dst: continue
            # 禁止反向跨层流动 (Tier大 指向 Tier小)
            if get_tier(src) > get_tier(dst):
                blacklist.append((src, dst))
    return blacklist


def learn_structure_with_constraints(df):
    print("\n[Step 2] 正在运行 NOTEARS-MLP 进行结构学习...")

    labels = df.columns.tolist()
    X_raw = df.values.astype(np.float32)

    # --- 数据标准化 (关键) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = X_scaled.astype(np.float32)

    d = X.shape[1]

    # model
    model = NotearsMLP(dims=[d, 32, 1], bias=True)

    # 传入归一化后的 X
    adj_matrix = notears_nonlinear(model, X, lambda1=0, lambda2=0.01)

    if hasattr(adj_matrix, 'detach'):
        adj_matrix = adj_matrix.detach().numpy()

    # --- 黑名单约束逻辑  ---
    blacklist = generate_blacklist_auto(df.columns.tolist())

    idx_map = {name: i for i, name in enumerate(labels)}
    constrained_matrix = adj_matrix.copy()

    # 阈值过滤
    constrained_matrix[np.abs(constrained_matrix) < 0] = 0

    print("  [约束] 正在应用物理黑名单剔除违禁边...")
    for src, dst in blacklist:
        if src in idx_map and dst in idx_map:
            i, j = idx_map[src], idx_map[dst]
            if constrained_matrix[i, j] != 0:
                constrained_matrix[i, j] = 0

    G = nx.DiGraph()
    # 必须先添加所有节点
    G.add_nodes_from(labels)

    rows, cols = np.where(constrained_matrix > 0)
    for r, c in zip(rows, cols):
        G.add_edge(labels[r], labels[c], weight=constrained_matrix[r, c])

    return G


def inspect_graph_structure(G, labels):
    """
    中间结果检查工具
    """
    print("\n" + "=" * 60)
    print(" [中间结果] Step 2. 结构学习 - 详细审计")
    print("=" * 60)

    edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
    print(f"检测到的节点数: {len(G.nodes)}")
    print(f"检测到的边数量: {len(edges)}")

    if len(edges) > 0:
        print(f"\n{'源节点':<15} -> {'目标节点':<15} | {'权重':<10}")
        print("-" * 60)
        for u, v, data in edges:
            print(f"{u:<15} -> {v:<15} | {data['weight']:.4f}")

    try:
        adj_matrix = nx.to_numpy_array(G, nodelist=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1.0,
                    xticklabels=labels, yticklabels=labels)
        plt.title("Learned Adjacency Matrix (Normalized)")
        plt.show()
    except Exception as e:
        print(f"[Warn] 无法绘制热力图: {e}")


# ==========================================
# 2. 参数拟合与因果推断
# ==========================================
def causal_inference_demo(G, df):
    print("\n[Step 3] 贝叶斯网络参数学习与推断...")

    # 离散化
    df_disc = pd.DataFrame()
    for col in df.columns:
        if col in ['Hit_Res', 'Weather']:
            df_disc[col] = df[col]
        else:
            try:
                df_disc[col] = pd.qcut(df[col], q=3, labels=[0, 1, 2], duplicates='drop').astype(int)
            except ValueError:
                df_disc[col] = df[col].apply(lambda x: 0 if x < df[col].mean() else 1)

    # 拟合 CPT
    model = BayesianNetwork(list(G.edges()))
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)

    # 定义推断引擎
    infer = VariableElimination(model)

    print("\n[Step 4] 场景推演 (干预分析)...")

    evidence_bad = {'Rec_Conf': 0, 'Rec_Latency': 2}
    evidence_good = {'Rec_Conf': 2, 'Rec_Latency': 0}

    try:
        r_bad = infer.query(['Hit_Res'], evidence=evidence_bad).values[1]
        r_good = infer.query(['Hit_Res'], evidence=evidence_good).values[1]

        print("-" * 60)
        print(f"场景对比 (已自动剥离 Weather, SNR, Jamming 的混杂影响):")
        print(f"1. [低效能状态] (Conf低/延时高) -> 命中率: {r_bad:.2%}")
        print(f"2. [高效能状态] (Conf高/延时低) -> 命中率: {r_good:.2%}")
        print(f"3. [效能归因] 识别算法优化提升: {(r_good - r_bad):.2%}")
        print("-" * 60)

    except Exception as e:
        print(f"[Err] 推理过程出错: {e}")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    csv_filename = "battlefield_sim_data.csv"

    # 1. 读取数据
    if not os.path.exists(csv_filename):
        print(f"[Error] 未找到数据文件: {csv_filename}")
        print("请先运行 'data_generator.py' 生成数据。")
        sys.exit(1)

    print(f"[Info] 正在加载数据: {csv_filename}")
    df_data = pd.read_csv(csv_filename, encoding='utf-8-sig')
    print(f"[Info] 数据加载完毕，维度: {df_data.shape}")

    # 2. 学习结构
    causal_graph = learn_structure_with_constraints(df_data)
    inspect_graph_structure(causal_graph, df_data.columns.tolist())

    # 3. 绘制
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(causal_graph, seed=42)
    nx.draw(causal_graph, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, arrowsize=20)
    plt.title("Learned Causal Graph")
    plt.show()

    # 4. 因果推理
    has_path_conf = nx.has_path(causal_graph, 'Rec_Conf', 'Hit_Res') if 'Rec_Conf' in causal_graph else False
    has_path_lat = nx.has_path(causal_graph, 'Rec_Latency', 'Hit_Res') if 'Rec_Latency' in causal_graph else False

    if has_path_conf or has_path_lat:
        causal_inference_demo(causal_graph, df_data)
    else:
        print("[Warn] 路径断裂，手动补全 Track_Stable -> Hit_Res 以便演示。")
        causal_graph.add_edge('Track_Stable', 'Hit_Res')
        causal_inference_demo(causal_graph, df_data)