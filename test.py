import math

# === 全局参数（可以视为常数） ===
PHI0 = 3.0  # 安静日基准通量 Phi0
A = 4.0  # 纬度增强系数 a
BETA = 1.5  # 纬度指数 beta
C = 0.2  # 高度系数 c
H0 = 550.0  # 参考高度 h0 (km)
SIGMA_EFF = 1e-7  # 有效 SEU 截面 sigma_eff
DT = 1.0  # 时间步长 Δt (s)


def f_lat(lat_deg: float) -> float:
    """
    纬度因子 f_lat(λ_m) = 1 + a*(|λ_m|/90)^β
    这里用地理纬度近似地磁纬度，lat_deg 取 [-90, 90]
    """
    return 1.0 + A * (abs(lat_deg) / 90.0) ** BETA


def f_alt(h_km: float) -> float:
    """
    高度因子 f_alt(h) = 1 + c*(h - h0)/h0
    h_km 单位：km
    """
    return 1.0 + C * (h_km - H0) / H0


def p_seu(lat_deg: float, h_km: float, S: float) -> float:
    """
    计算某颗卫星在给定风暴强度 S 下，
    当前时间步内发生“致命 SEU”的概率 p_SEU,i(t)

    S：风暴强度，表示通量相对安静日的倍数
    """
    Phi_i = PHI0 * S * f_lat(lat_deg) * f_alt(h_km)
    lam_i = Phi_i * SIGMA_EFF
    return 1.0 - math.exp(-lam_i * DT)


def p_fail_isl(lat_i: float, h_i: float,
               lat_j: float, h_j: float,
               S: float) -> float:
    """
    计算星间链路(i, j)在当前时间步内的失效概率 p_fail,ij(t)
    假设两端 SEU 独立：
    p_fail,ij = 1 - (1 - p_SEU,i) * (1 - p_SEU,j)
    """
    p_i = p_seu(lat_i, h_i, S)
    p_j = p_seu(lat_j, h_j, S)
    return 1.0 - (1.0 - p_i) * (1.0 - p_j)


S = 500        # 风暴强度

latA, hA = 0.0, 550.0
latB, hB = 60.0, 550.0

pA = p_seu(latA, hA, S)
pB = p_seu(latB, hB, S)
pAB = p_fail_isl(latA, hA, latB, hB, S)

print("p_SEU_A =", pA)
print("p_SEU_B =", pB)
print("p_fail_AB =", pAB)
