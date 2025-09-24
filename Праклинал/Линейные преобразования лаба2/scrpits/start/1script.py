import os

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
def print_matrix_pretty(M):
    print(f"({M[0,0]:.3f}, {M[0,1]:.3f})")
    print(f"({M[1,0]:.3f}, {M[1,1]:.3f})")
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

def eigvals_text(M):
    w = np.linalg.eigvals(M)
    w = np.sort_complex(w)
    parts = []
    for z in w:
        if abs(z.imag) < 1e-10:
            parts.append(f"{z.real:.6g}")
        else:
            parts.append(f"{z.real:.6g}{'+' if z.imag>=0 else '-'}{abs(z.imag):.6g}i")
    return parts

def save_explanation(filename, title, M, det_txt, eig_txt, reasoning, params_note):
    os.makedirs("explanations", exist_ok=True)
    with open(os.path.join("explanations", filename), "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("**Матрица (построчно):**\n\n")
        f.write(f"`({M[0,0]:.6g}, {M[0,1]:.6g})`\n\n`({M[1,0]:.6g}, {M[1,1]:.6g})`\n\n")
        f.write(f"**det:** `{det_txt}`\n\n")
        f.write(f"**Собственные значения:** `{', '.join(eig_txt)}`\n\n")
        f.write("**Рассуждение / теоретическое обоснование:**\n\n")
        f.write(reasoning.strip() + "\n\n")
        f.write(params_note + "\n")


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

common_note = ("Приведен вывод, расчёты и теоретическое обоснование полученных матриц. "
               "Попытайтесь найти наиболее общий класс матриц, прежде чем давать частное решение.")

# краткие «титулы» для файлов и развёрнутые рассуждения
explanations = {
    "1": {
        "title": "1) Отражение относительно y=ax",
        "slug": "01_reflect_y=ax",
        "reason": (
            "Искомый класс — все ортогональные преобразования с det=-1, отражающие относительно прямой под углом φ к Ox. "
            "В базисе, где ось совпадает с линией y=ax (т.е. после поворота на φ=arctan(a)), отражение — diag(1,-1). "
            "Переход назад: S(φ)=R(φ)·diag(1,-1)·R(-φ) ⇒ формула с дробями. det(S)=-1, спектр {1,-1}: "
            "направление прямой — λ=1, перпендикуляр — λ=-1."
        )},
    "2": {
        "title": "2) Проекция на y=bx",
        "slug": "02_proj_y=bx",
        "reason": (
            "Искомый класс — все ортогональные проекторы ранга 1 на прямую под углом ψ=arctan(b). "
            "В повёрнутом базисе матрица проекции — diag(1,0). Возврат поворотом даёт стандартную формулу. "
            "det=0 (ранг 1), спектр {1,0}: вдоль линии сохраняет, на ортогональ зануляет."
        )},
    "3": {
        "title": "3) Поворот на 10c°",
        "slug": "03_rotate_10c",
        "reason": (
            "Класс — все ортогональные преобразования с det=+1 (повороты). "
            "Матрица R(α) = [[cosα, -sinα],[sinα, cosα]]. Для α=10c° получаем дет=1, комплексный спектр e^{±iα} "
            "(вещественные СВ только при α≡0,π)."
        )},
    "4": {
        "title": "4) Центральная симметрия",
        "slug": "04_central_symmetry",
        "reason": (
            "Симметрия относительно начала координат — это масштабирование на -1: A=-I. "
            "Класс — все скалярные матрицы λI с λ=-1. det=(-1)^2=1, спектр {-1,-1}."
        )},
    "5": {
        "title": "5) R(-10d°) после отражения y=ax",
        "slug": "05_rotate_after_reflect",
        "reason": (
            "Композиция отражения (det=-1) и поворота (det=+1) даёт общее ортогональное преобразование с det=-1. "
            "Такое преобразование — отражение относительно некоторой прямой: спектр {1,-1}. "
            "Формула A=R(-10d°)·S_a описывает общий класс всех отражений, сдвинутых поворотом."
        )},
    "6": {
        "title": "6) e1→(1,a), e2→(1,b)",
        "slug": "06_basis_images",
        "reason": (
            "Искомый класс — все линейные операторы, однозначно задаваемые образами базисных векторов. "
            "Столбцы матрицы — образы e1 и e2, поэтому A=[[1,1],[a,b]]. det=b-a (≠0 при a≠b). "
            "Собственные значения находятся из χ(λ)=λ^2-(1+b)λ+(b-a)."
        )},
    "7": {
        "title": "7) Обратное к п.6",
        "slug": "07_inverse_of_6",
        "reason": (
            "Класс — обратимые преобразования к (6): A7=A6^{-1}. det(A7)=1/det(A6). "
            "Спектр обращается покомпонентно: λ_i(A7)=1/λ_i(A6)."
        )},
    "8": {
        "title": "8) Поменять местами прямые y=ax и y=bx",
        "slug": "08_swap_lines",
        "reason": (
            "Нужен оператор, который в базисе u=(1,a) и v=(1,b) действует как перестановка векторов: [[0,1],[1,0]]. "
            "Переход к стандартному базису даёт A=[v u][u v]^{-1}. det=-1 (чистая инволюция), спектр {1,-1}."
        )},
    "9": {
        "title": "9) Круг→круг площади c (недиагональная)",
        "slug": "09_circle_to_circle",
        "reason": (
            "Общий класс отображений «круг→круг» — ортогональный поворот/отражение + одинаковый масштаб s. "
            "Площадь умножается на det(A)=s^2, значит s=√c. Берём поворот на 45°, чтобы матрица была недиагональна: A=√c·R(45°). "
            "Спектр: √c·e^{±i45°}."
        )},
    "10": {
        "title": "10) Круг→не-круг площади d (недиагональная)",
        "slug": "10_circle_to_ellipse",
        "reason": (
            "Чтобы «круг→не-круг», нужна неортогональная матрица (разные растяжения/сдвиг). "
            "Пример общего класса — верхнетреугольные [[d,1],[0,1]]: det=d (площадь d), недиагональная при ненулевом внедиагональном элементе. "
            "Собственные значения — диагональные элементы: {d,1}."
        )},
    "11": {
        "title": "11) Перпендикулярные СВ, не на y=0 и не на y=x",
        "slug": "11_symmetric_rotated",
        "reason": (
            "Общий класс — симметричные матрицы с различными λ: они ортогонально диагонализуемы, собственные векторы взаимно перпендикулярны. "
            "Чтобы направления не совпадали с осями/диагональю, берём поворот на θ=arctan(a)≠0,π/4. "
            "A=R(θ)·diag(λ1,λ2)·R(-θ) (λ1≠λ2). det=λ1λ2, спектр {λ1,λ2}."
        )},
    "12": {
        "title": "12) Нет двух неколлинеарных СВ (дефектная)",
        "slug": "12_defective_jordan",
        "reason": (
            "Класс — жордановы блоки 2×2: [[λ,1],[0,λ]]. Один собственный вектор, геометрическая кратность 1. "
            "det=λ^2 (у нас 9), алгебраический спектр {λ,λ}. Вещественно не диагонализуется."
        )},
    "13": {
        "title": "13) Вещественная без вещественных СВ",
        "slug": "13_real_no_real_eigs",
        "reason": (
            "Класс — чистые повороты на угол ≠0,π: R(α) с det=1. Для α=90° собственные значения комплексные ±i. "
            "Ни одного вещественного собственного вектора."
        )},
    "14": {
        "title": "14) Любой ненулевой вектор — собственный",
        "slug": "14_scalar_matrix",
        "reason": (
            "Единственный класс — скалярные матрицы λI. Любой v≠0 удовлетворяет Av=λv. "
            "det=λ^2, спектр {λ,λ}. В примере λ=-2."
        )},
    "15A": {
        "title": "15A) A и 15B) B с AB≠BA: A",
        "slug": "15A_noncomm_A",
        "reason": (
            "Берём сдвиг A=[[1,1],[0,1]] — жорданов блок с λ=1 (дефектная). "
            "Он не коммутирует с поворотом: векторное действие нарушает порядок операций."
        )},
    "15B": {
        "title": "15B) B (для AB≠BA): поворот на 90°",
        "slug": "15B_noncomm_B",
        "reason": (
            "Поворот B=R(90°) — ортогональный оператор det=1, спектр ±i. "
            "Композиции A·B и B·A дают разные матрицы, хотя det и tr совпадают ⇒ AB≠BA."
        )},
    "15AB": {
        "title": "15AB) Произведение A·B",
        "slug": "15AB_product",
        "reason": (
            "Конкретное произведение демонстрирует некоммутативность: AB≠BA. "
            "Спектр совпадает (подобные многочлены могут совпадать), но матрицы различны."
        )},
    "15BA": {
        "title": "15BA) Произведение B·A",
        "slug": "15BA_product",
        "reason": (
            "Аналогично пункту 15AB, но обратный порядок. Матрицы разные при равных det и tr — наглядный контрпример коммутативности."
        )},
    "16A": {
        "title": "16A) A и 16B) B с AB=BA: A",
        "slug": "16A_comm_A",
        "reason": (
            "Общий класс коммутирующих — одновременно диагонализуемые одним ортогональным базисом. "
            "Берём A=R(θ)·diag(2,3)·R(-θ), B с тем же R(θ) ⇒ общие собственные векторы, значит AB=BA."
        )},
    "16B": {
        "title": "16B) B (для AB=BA)",
        "slug": "16B_comm_B",
        "reason": (
            "Та же конструкция, но с другими собственными числами (5,7). "
            "Оба оператора имеют общий набор собственных направлений — отсюда коммутативность."
        )}
}

# ===== печать + сохранение рассуждений =====
# Соберём единый список отображений с описаниями (как в прошлой версии)
items = [
    (A1,  "1"),
    (A2,  "2"),
    (A3,  "3"),
    (A4,  "4"),
    (A5,  "5"),
    (A6,  "6"),
    (A7,  "7"),
    (A8,  "8"),
    (A9,  "9"),
    (A10, "10"),
    (A11, "11"),
    (A12, "12"),
    (A13, "13"),
    (A14, "14"),
]
pairs = [
    (A15A, "15A"),
    (A15B, "15B"),
    (A15AB,"15AB"),
    (A15BA,"15BA"),
    (A16A, "16A"),
    (A16B, "16B"),
]

params_note = f"*Числовая подстановка:* a={a}, b={b}, c={c}, d={d}."

for M, key in items + pairs:
    meta = explanations[key]
    det_val = np.linalg.det(M)
    eig_txt = eigvals_text(M)

    print(f"\n=== {meta['title']} ===")
    print_matrix_pretty(M)
    print(f"det = {det_val:.6g}")
    print("eigenvalues =", ", ".join(eig_txt))
    print("Обоснование:", meta["reason"])

    # сохранить в файл
    filename = f"{meta['slug']}.md"
    save_explanation(filename, meta["title"], M, f"{det_val:.6g}", eig_txt, meta["reason"], params_note)