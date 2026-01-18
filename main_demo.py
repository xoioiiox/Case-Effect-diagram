import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# pgmpy 库用于参数学习和推理
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# --- 0. NOTEARS 导入检查 ---
# 尝试导入 notears，如果不存在则使用 Mock 函数以便代码演示
try:
    sys.path.append("./notears")  # 假设你的仓库在子目录
    from notears.nonlinear import notears_nonlinear, NotearsMLP

    print("[INFO] 成功导入 notears_nonlinear 库")
except ImportError:
    print("[WARN] 未找到 notears 库，正在使用模拟生成的邻接矩阵进行演示...")


    # 这是一个 Mock 函数，仅用于演示流程，实际请确保库路径正确
    def notears_nonlinear(X, lambda1, lambda2):
        d = X.shape[1]
        # 返回一个随机但稀疏的矩阵，模拟学习到的结构
        return np.random.uniform(0, 0.5, (d, d)) * (np.random.rand(d, d) > 0.7)


# ==========================================
# 1. 复杂场景数据构造 (Data Generation)
# ==========================================
def generate_battlefield_data(n=1000):
    """
    生成符合物理规律和战术逻辑的仿真数据 (修正版)
    """
    np.random.seed(42)

    # --- A. 强混杂因子 (Confounders) ---
    # Weather: 0=好天气, 1=坏天气. Beta分布模拟"大部分时候天气尚可"
    weather = np.random.beta(2, 5, n)

    # Jamming: 电磁干扰
    jamming = np.random.beta(1, 3, n)  # 0~1 之间，偏向低干扰但有高干扰尾部

    # Occlusion: 遮挡比例
    occlusion = np.random.beta(1, 3, n)

    # --- B. 中间物理量 ---
    # SNR: 确保信噪比范围合理 (0 ~ 30 dB)
    # 逻辑: 天气差衰减大，干扰强衰减大
    snr = 25 * np.exp(-2 * weather) - 15 * jamming + np.random.normal(0, 2, n)
    snr = np.clip(snr, 0, 30)

    # --- C. 干预变量 (Treatments - 识别算法表现) ---
    # Rec_Conf: 识别置信度
    base_conf = 0.5 + (snr / 60) - (occlusion * 0.4)
    rec_conf = base_conf + np.random.normal(0, 0.05, n)
    rec_conf = np.clip(rec_conf, 0.1, 0.99)

    # Rec_Latency: 识别耗时，干扰越大，算法处理越慢 (ms)
    rec_latency = 30 + (jamming * 40) + np.random.normal(0, 5, n)

    # --- D. 中间状态变量 (Mediators) ---
    # Track_Stable: 打击稳定性，受到识别准确性影响
    track_stable = 0.8 * rec_conf + np.random.normal(0, 0.05, n)
    track_stable = np.clip(track_stable, 0, 1)

    # Lock_Time: 锁定时间
    # 逻辑: 延迟高、跟踪不稳 -> 锁定变慢
    lock_time = 2.0 + (rec_latency / 50) - (track_stable * 2.0)
    lock_time = np.clip(lock_time, 0.5, 10.0)

    # --- E. 结果变量 (Outcome) ---
    # Hit_Score: 命中得分
    hit_score = 5 * track_stable - 0.5 * lock_time + np.random.normal(0, 0.5, n)

    # 转换为概率 (Sigmoid)
    hit_prob = 1 / (1 + np.exp(-(hit_score - 0.5)))  # 偏移一点确保正负样本平衡
    hit_res = np.random.binomial(1, hit_prob)

    df = pd.DataFrame({
        'Weather': weather,
        'Jamming': jamming,
        'Occlusion': occlusion,
        'SNR': snr,
        'Rec_Conf': rec_conf,
        'Rec_Latency': rec_latency,
        'Track_Stable': track_stable,
        'Lock_Time': lock_time,
        'Hit_Score': hit_score,
        'Hit_Res': hit_res
    })

    return df


# ==========================================
# 2. 结构学习与黑名单约束
# ==========================================
def learn_structure_with_constraints(df):
    print("\n[Step 2] 正在运行 NOTEARS-MLP 进行结构学习...")

    labels = df.columns.tolist()
    # 强制转换为 float32，这是 PyTorch 默认接受的类型
    X = df.values.astype(np.float32)
    d = X.shape[1]  # 变量维度

    # --- [修复点] 实例化 MLP 模型 ---
    # notears-mlp 需要先定义神经网络结构
    # dims = [输入维度, 隐藏层维度, 输出维度(通常为1)]
    # 隐藏层设为 10 或 20 即可满足 Demo 需求
    model = NotearsMLP(dims=[d, 10, 1], bias=True)

    # --- [修复点] 调用函数时传入 model ---
    # 注意参数顺序：(model, X, lambda1, lambda2)
    adj_matrix = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)

    # 如果返回的是 torch.Tensor，转为 numpy
    if hasattr(adj_matrix, 'detach'):
        adj_matrix = adj_matrix.detach().numpy()

    # --- 黑名单约束逻辑  ---
    blacklist = generate_blacklist_auto(df)

    idx_map = {name: i for i, name in enumerate(labels)}
    constrained_matrix = adj_matrix.copy()
    constrained_matrix[np.abs(constrained_matrix) < 0.1] = 0  # 阈值过滤

    print("  [约束] 正在应用物理黑名单剔除违禁边...")
    for src, dst in blacklist:
        if src in idx_map and dst in idx_map:
            i, j = idx_map[src], idx_map[dst]
            if constrained_matrix[i, j] != 0:
                constrained_matrix[i, j] = 0

    G = nx.DiGraph()
    rows, cols = np.where(constrained_matrix > 0)
    for r, c in zip(rows, cols):
        G.add_edge(labels[r], labels[c], weight=constrained_matrix[r, c])

    return G


def generate_blacklist_auto(df_columns):
    """
    根据变量层级自动生成黑名单
    """
    blacklist = []

    # 1. 定义每个变量所属的层级 (支持模糊匹配/前缀)
    # 没在字典里的默认设为中间层，或者你可以设个默认值
    tier_map = {
        # Tier 0: 环境/输入 (绝对上游)
        'Weather': 0, 'Jamming': 0, 'Occlusion': 0,

        # Tier 1: 传感器/物理量
        'SNR': 1,

        # Tier 2: 算法/控制
        'Rec_Conf': 2, 'Rec_Latency': 2,

        # Tier 3: 状态/过程
        'Track_Stable': 3, 'Lock_Time': 3,

        # Tier 4: 结果 (绝对下游)
        'Hit_Res': 4
    }

    # 辅助函数：获取列名的层级
    def get_tier(col_name):
        # 优先精确匹配
        if col_name in tier_map:
            return tier_map[col_name]
        # 其次前缀/模糊匹配 (适合这100个特征)
        for key, level in tier_map.items():
            if key in col_name:  # 比如 'Sensor_01', 'Sensor_02' 都会匹配到 'Sensor_'
                return level
        return 2  # 默认层级 (比如算法层)

    # 2. 遍历所有两两组合，自动生成禁忌边
    # 复杂度是 O(N^2)，对于 100 个变量也就 10000 次循环，非常快
    for src in df_columns:
        for dst in df_columns:
            if src == dst: continue

            src_tier = get_tier(src)
            dst_tier = get_tier(dst)

            # 规则：禁止反向跨层流动
            # 如果源节点层级 > 目标节点层级，说明是反向边 (例如 结果->原因)，禁止！
            if src_tier > dst_tier:
                blacklist.append((src, dst))

            # 可选规则：禁止同一层级内乱指 (如果你希望层内独立)
            # if src_tier == dst_tier: blacklist.append((src, dst))

    print(f"[自动化黑名单] 根据层级逻辑，已自动生成 {len(blacklist)} 条禁忌边。")
    return blacklist

import seaborn as sns  # 如果没有安装 seaborn，可以用 matplotlib 代替，但 seaborn 更美观


def inspect_graph_structure(G, labels):
    """
    中间结果检查工具：打印边列表并绘制热力图
    """
    print("\n" + "=" * 60)
    print(" [中间结果] Step 2. 结构学习 - 详细审计")
    print("=" * 60)

    # 1. 文本输出：打印所有边和权重
    edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)

    print(f"检测到的节点数: {len(G.nodes)}")
    print(f"检测到的边数量: {len(edges)}")

    if len(edges) > 0:
        print("\n--- 发现的因果链路 (按权重降序) ---")
        print(f"{'源节点 (Cause)':<15} -> {'目标节点 (Effect)':<15} | {'权重 (Weight)':<10}")
        print("-" * 60)
        for u, v, data in edges:
            print(f"{u:<15} -> {v:<15} | {data['weight']:.4f}")
    else:
        print("\n[警告] ⚠️ 未发现任何边！建议减小 lambda1 (稀疏惩罚) 或降低阈值。")
    print("=" * 60 + "\n")

    # 2. 图形输出：邻接矩阵热力图 (Adjacency Matrix Heatmap)
    # 这能帮你一眼看出是否有“反直觉”的连接
    try:
        adj_matrix = nx.to_numpy_array(G, nodelist=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Learned Adjacency Matrix (Weights)")
        plt.ylabel("Source (Cause)")
        plt.xlabel("Target (Effect)")
        plt.tight_layout()
        plt.show()  # 这会弹出一个窗口
        print("[Info] 热力图已生成，请检查窗口。")
    except Exception as e:
        print(f"[Warn] 无法绘制热力图 (可能缺少 seaborn): {e}")

# ==========================================
# 3. 参数拟合与因果推断
# ==========================================
def causal_inference_demo(G, df):
    print("\n[Step 3] 贝叶斯网络参数学习与推断...")

    # 3.1 离散化 (Discretization)
    # pgmpy 的推理在离散数据上最稳定。我们将连续变量分为 3 档 (Low/Med/High)
    df_disc = pd.DataFrame()
    for col in df.columns:
        if col == 'Hit_Res':
            df_disc[col] = df[col]  # 已经是0/1，无需切分
        else:
            # qcut 按分位数切分，保证每箱样本均衡
            try:
                df_disc[col] = pd.qcut(df[col], q=3, labels=[0, 1, 2]).astype(int)
            except ValueError:  # 防止某些列数值单一报错
                df_disc[col] = df[col].apply(lambda x: 0 if x < 0.5 else 1)

    # 3.2 拟合 CPT
    model = BayesianNetwork(list(G.edges()))
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)

    # 3.3 定义推断引擎
    infer = VariableElimination(model)

    print("\n[Step 4] 场景推演 (干预分析)...")

    # --- 场景定义 ---
    # 我们不仅关注置信度，还关注时延和IOU的综合影响

    # 场景 A: 算法表现不佳 (Confidence低, IoU低, Latency高)
    # 对应离散值: Conf=0, IoU=0, Latency=2 (注意Latency越大越差)
    evidence_bad = {'Rec_Conf': 0, 'BBox_IoU': 0, 'Rec_Latency': 2}

    # 场景 B: 算法表现卓越 (Confidence高, IoU高, Latency低)
    # 对应离散值: Conf=2, IoU=2, Latency=0
    evidence_good = {'Rec_Conf': 2, 'BBox_IoU': 2, 'Rec_Latency': 0}

    try:
        # 计算 P(Hit_Res=1 | Evidence)
        r_bad = infer.query(['Hit_Res'], evidence=evidence_bad).values[1]
        r_good = infer.query(['Hit_Res'], evidence=evidence_good).values[1]

        print("-" * 60)
        print(f"场景对比 (已自动剥离 Weather, SNR, Jamming 的混杂影响):")
        print(f"1. [低效能状态] (Conf低/IoU低/延时高) -> 命中率: {r_bad:.2%}")
        print(f"2. [高效能状态] (Conf高/IoU高/延时低) -> 命中率: {r_good:.2%}")
        print(f"3. [效能归因] 识别算法全维度优化可带来的战果提升: {(r_good - r_bad):.2%}")
        print("-" * 60)

        # 进一步细分：只优化 Latency 会怎样？(控制变量法)
        # 假设置信度中等，只改变时延
        r_lat_high = infer.query(['Hit_Res'], evidence={'Rec_Conf': 1, 'Rec_Latency': 2}).values[1]
        r_lat_low = infer.query(['Hit_Res'], evidence={'Rec_Conf': 1, 'Rec_Latency': 0}).values[1]
        print(f"4. [单变量分析] 仅优化处理时延 (High->Low) 的贡献: {(r_lat_low - r_lat_high):.2%}")

    except Exception as e:
        print(f"[Err] 推理过程出错，可能是生成的图结构不连通导致: {e}")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 生成数据
    df_data = generate_battlefield_data()
    print(f"[Info] 数据生成完毕: {df_data.shape}")
    # --- [新增] 保存为 CSV 文件 ---
    # index=False 表示不保存行号(0,1,2...)
    # encoding='utf-8-sig' 确保 Excel 打开中文不乱码
    csv_filename = "battlefield_sim_data.csv"
    df_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"[Info] 已将生成的数据保存至: {csv_filename}")

    # 2. 学习结构 (带黑名单)
    causal_graph = learn_structure_with_constraints(df_data)
    inspect_graph_structure(causal_graph, df_data.columns.tolist())

    # 3. 绘制简单图谱
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(causal_graph, seed=42)
    nx.draw(causal_graph, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, arrowsize=20)
    plt.title("Learned Causal Graph (NOTEARS + Blacklist)")
    # plt.show() # 如果在本地环境运行，请取消注释

    # 4. 因果推理
    # 检查是否生成了从 识别 -> 命中 的路径，如果有才进行推理
    if nx.has_path(causal_graph, 'Rec_Conf', 'Hit_Res'):
        causal_inference_demo(causal_graph, df_data)
    else:
        print("[Warn] NOTEARS 未发现从算法指标到命中率的直接或间接路径，尝试降低 lambda1 阈值。")
        # 强制添加边以演示推理 (仅用于 Demo)
        causal_graph.add_edge('Rec_Conf', 'Track_Stable')
        causal_graph.add_edge('Track_Stable', 'Hit_Res')
        causal_inference_demo(causal_graph, df_data)