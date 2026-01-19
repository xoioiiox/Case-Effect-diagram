# filename: data_generator.py
import numpy as np
import pandas as pd

def generate_battlefield_data(n=3000):
    """
    生成符合物理规律和战术逻辑的仿真数据 (修正版)
    """
    np.random.seed(42)

    # --- A. 强混杂因子 (Confounders) ---
    # 1. Weather (天气)
    # 连续值用于物理计算 (0=极好, 1=极差)
    weather_continuous = np.random.beta(2, 5, n)
    # 离散值用于因果推断 (0:晴朗, 1:多云, 2:恶劣)
    weather = np.digitize(weather_continuous, bins=[0.3, 0.6])

    # 2. Jamming (电磁干扰) - U型分布
    jamming = np.random.beta(0.5, 0.5, n)

    # 3. Occlusion (遮挡) - 均匀分布
    occlusion = np.random.uniform(0, 0.6, n)

    # --- B. 中间物理量 ---
    # [关键修正] SNR: 确保逻辑是"天气越差SNR越低"
    # 基础SNR 30dB, 随天气指数衰减, 随干扰线性减小
    snr = 30 * np.exp(-2 * weather) - 15 * jamming
    snr = np.clip(snr, 0, 30)

    # --- C. 干预变量 (Treatments) ---
    # Rec_Conf: 识别置信度
    base_conf = 0.5 + (snr / 60) - (occlusion * 0.4)
    rec_conf = np.clip(base_conf, 0.1, 0.99)

    # Rec_Latency: 识别耗时 (ms)
    rec_latency = 30 + (jamming * 40)

    # --- D. 中间状态变量 (Mediators) ---
    # Track_Stable: 跟踪稳定性
    track_stable = 0.8 * rec_conf
    track_stable = np.clip(track_stable, 0, 1)

    # Lock_Time: 锁定时间
    lock_time = 2.0 + (rec_latency / 50) - (track_stable * 2.0)
    lock_time = np.clip(lock_time, 0.5, 10.0)

    # --- E. 结果变量 (Outcome) ---
    # Hit_Score: 命中得分
    hit_score = 5 * track_stable - 0.5 * lock_time - 1.0 * weather

    # 转换为概率 (包含硬阈值截断)
    raw_prob = 1 / (1 + np.exp(-hit_score))
    final_prob = np.where(hit_score < 0, 0, raw_prob)
    hit_res = np.random.binomial(1, final_prob)

    df = pd.DataFrame({
        'Weather': weather,
        'Jamming': jamming,
        'Occlusion': occlusion,
        'SNR': snr,
        'Rec_Conf': rec_conf,
        'Rec_Latency': rec_latency,
        'Track_Stable': track_stable,
        'Lock_Time': lock_time,
        'Hit_Res': hit_res
    })

    return df

if __name__ == "__main__":
    # 生成数据
    print("[Info] 正在生成仿真数据...")
    df_data = generate_battlefield_data()
    print(f"[Info] 数据生成完毕，维度: {df_data.shape}")

    # 保存文件
    csv_filename = "battlefield_sim_data.csv"
    df_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"[Success] 数据已保存至: {csv_filename}")