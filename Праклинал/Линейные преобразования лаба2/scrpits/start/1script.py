import numpy as np
import matplotlib.pyplot as plt

def draw_axes(ax, lim=10):
    """Рисует декартову систему координат."""
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)

def plot_polygon(ax, points, color, label):
    """Строит многоугольник по точкам."""
    polygon = np.vstack([points, points[0]])  # замыкаем
    ax.plot(polygon[:,0], polygon[:,1],
            marker='o', markersize=6,
            linewidth=2, color=color, label=label)

def transform_polygon(points, matrix):
    """Умножает все точки многоугольника на матрицу 2x2."""
    return np.dot(points, matrix.T)

# === Исходные данные ===
points = np.array([
    [1,1],
    [4,1],
    [5,5],
    [-1,6],
    [-3,2]
])

# Пример матрицы (отражение относительно y=x)
matrix = np.array([
    [0, 1],
    [1, 0]
])

# === Применяем преобразование ===
transformed_points = transform_polygon(points, matrix)

# === Визуализация ===
fig, ax = plt.subplots(figsize=(6,6))  # квадратное окно
draw_axes(ax, lim=8)

plot_polygon(ax, points, color='royalblue', label='Исходный')
plot_polygon(ax, transformed_points, color='crimson', label='Преобразованный')

ax.legend(loc='upper left', fontsize=10, frameon=True)

# Сохраняем именно то изображение, что нарисовано
plt.savefig("polygon.jpg", format="jpg", dpi=300, bbox_inches="tight")

plt.show()
