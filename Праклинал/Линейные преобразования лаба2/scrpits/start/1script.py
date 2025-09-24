import numpy as np
import matplotlib.pyplot as plt

def draw_axes(ax, lim=10):
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)

def plot_polygon(ax, points, color, label):
    polygon = np.vstack([points, points[0]])
    ax.plot(polygon[:,0], polygon[:,1],
            marker='o', markersize=6,
            linewidth=2, color=color, label=label)

def transform_polygon(points, matrix):
    return np.dot(points, matrix.T)

def reflect_a(a):
    return (1/(1+a*a)) * np.array([[1-a*a, 2*a],
                                   [2*a,   a*a-1]])

def project_b(b):
    return (1/(1+b*b)) * np.array([[1, b],
                                   [b, b*b]])

def R_deg(theta_deg):
    th = np.deg2rad(theta_deg)
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th),  np.cos(th)]])

def draw_eig_lines(ax, M, lim=8):
    # Рисуем направления собственных векторов, если обе λ вещественные и разные
    w, v = np.linalg.eig(M)
    if np.all(np.isreal(w)):
        w = np.real(w)
        # если разные и не вырождены
        if abs(w[0] - w[1]) > 1e-9:
            for i in (0,1):
                vec = np.real(v[:, i])
                if np.linalg.norm(vec) < 1e-9:
                    continue
                vec = vec / np.linalg.norm(vec)
                # линия через 0 по направлению vec
                x = np.array([-lim, lim])
                y = (vec[1]/(vec[0]+1e-12)) * x
                ax.plot(x, y, linestyle='-', linewidth=1.5, alpha=0.7,
                        label=f"eig dir λ≈{w[i]:.3g}")

def print_matrix(M):
    for row in M:
        print("(" + ", ".join(f"{val:.3f}" for val in row) + ")")

def print_det_eigs(name, M):
    print(f"\n{name}:")
    print_matrix(M)
    det = np.linalg.det(M)
    w = np.linalg.eigvals(M)
    print(f"det = {det:.6g}")
    print("eigenvalues =", [f"{val:.3g}" for val in np.sort_complex(w)])

# === Исходные данные ===
points = np.array([
    [1,1],
    [4,1],
    [5,5],
    [-1,6],
    [-3,2]
])

# === Параметры ===
a, b, c, d = 2, -3, 4, 5

# === Набор матриц по пунктам 1–16 ===
theta_c = 10 * c     # 40°
theta_d = 10 * d     # 50°

theta_a = np.rad2deg(np.arctan(a))
theta_b = np.rad2deg(np.arctan(b))

A1  = reflect_a(a)
A2  = project_b(b)
A3  = R_deg(theta_c)
A4  = -np.eye(2)
A5  = R_deg(-theta_d) @ reflect_a(a)              # поворот по часовой на 10d°
A6  = np.array([[1, 1], [a, b]])
A7  = np.linalg.inv(A6)
# Перестановка прямых (инволюция в базисе u=(1,a), v=(1,b)):
# В каноническом базисе удобно взять A8 = [v u][u v]^{-1}
u = np.array([[1],[a]])
v = np.array([[1],[b]])
A8  = np.hstack([v, u]) @ np.linalg.inv(np.hstack([u, v]))

A9  = np.sqrt(c) * R_deg(45)
A10 = np.array([[d, 1], [0, 1]])
# 11: симметричная с повёрнутыми собственными осями (перпендикулярные)
A11 = R_deg(theta_a) @ np.diag([2, 7]) @ R_deg(-theta_a)
# 12: дефектная Жорданова
A12 = np.array([[3, 1], [0, 3]])
# 13: поворот на 90°
A13 = R_deg(90)
# 14: скалярная
A14 = -2 * np.eye(2)
# 15: некоммутирующие
A15A = np.array([[1, 1], [0, 1]])
A15B = R_deg(90)
A15AB = A15A @ A15B
A15BA = A15B @ A15A
# 16: коммутирующие (совместно диагонализуемые)
A16A = R_deg(theta_b) @ np.diag([2, 3]) @ R_deg(-theta_b)
A16B = R_deg(theta_b) @ np.diag([5, 7]) @ R_deg(-theta_b)

items = [
    (A1,  "1) отражение относительно y=2x"),
    (A2,  "2) проекция на y=-3x"),
    (A3,  "3) поворот на 40°"),
    (A4,  "4) центральная симметрия"),
    (A5,  "5) R(-50°) после отражения y=2x"),
    (A6,  "6) e1→(1,2), e2→(1,-3)"),
    (A7,  "7) обратная к пункту 6"),
    (A8,  "8) перестановка прямых y=2x ↔ y=-3x"),
    (A9,  "9) 2·R(45°) (круг→круг, площадь 4)"),
    (A10, "10) [[5,1],[0,1]] (круг→эллипс, площадь 5)"),
    (A11, "11) симметричная с повёрнутыми осями"),
    (A12, "12) дефектная Жорданова"),
    (A13, "13) поворот на 90°"),
    (A14, "14) скалярная (-2I)"),
]

# + пары из 15 и 16
pairs = [
    (A15A, "15A) A = [[1,1],[0,1]]"),
    (A15B, "15B) B = R(90°)"),
    (A15AB,"15AB) A·B"),
    (A15BA,"15BA) B·A"),
    (A16A, "16A) A = R(atan(b))·diag(2,3)·R(-atan(b))"),
    (A16B, "16B) B = R(atan(b))·diag(5,7)·R(-atan(b))"),
]

# Печать детерминантов и спектров
for M, desc in items + pairs:
    print_det_eigs(desc, M)

# ВИЗУАЛИЗАЦИЯ
for M, desc in items + pairs:
    transformed_points = transform_polygon(points, M)
    fig, ax = plt.subplots(figsize=(6,6))
    draw_axes(ax, lim=8)
    plot_polygon(ax, points, color='royalblue', label='Исходный')
    plot_polygon(ax, transformed_points, color='crimson', label={desc})
    # линии собственных векторов, если есть 2 разные вещественные λ
    try:
        draw_eig_lines(ax, M, lim=8)
    except np.linalg.LinAlgError:
        pass
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    # имя файла без пробелов/скобок
    fname = "polygon_" + desc.replace(" ", "_").replace("(", "").replace(")", "") + ".jpg"
    plt.savefig(fname, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()