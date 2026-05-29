# -*- coding: utf-8 -*-
"""Sinh 21 hinh giai thich Chapter 2 + phan-cong + gantt."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "giai-thich-ch2"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 01 - anh mau
img = np.zeros((64, 64))
img[16:48, 16:48] = 1.0
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap="gray", vmin=0, vmax=1)
ax.set_title("Anh mau: hinh chu nhat trang tren nen den", fontsize=11)
ax.axis("off")
save(fig, "01-anh-mau.png")

# 02 - DFT y nghia
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis("off")
ax.text(0.5, 0.85, "DFT bien anh (mien khong gian) -> pho (mien tan so)", ha="center", fontsize=13, fontweight="bold")
for x, t in [(0.08, "Anh f(x,y)\ncon pixel"), (0.38, "DFT\nF(u,v) phuc"), (0.68, "Pha + bien do\nmoi tan so")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.35), 0.22, 0.35, boxstyle="round", fc="#E8F4FD", ec="black"))
    ax.text(x + 0.11, 0.52, t, ha="center", va="center", fontsize=10)
for x1, x2 in [(0.30, 0.38), (0.60, 0.68)]:
    ax.annotate("", xy=(x2, 0.52), xytext=(x1, 0.52), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(0.5, 0.12, "Tan so thap = vung muot | Tan so cao = bien canh, chi tiet", ha="center", fontsize=10)
save(fig, "02-dft-y-nghia.png")

# 03 - twiddle DFT1
N = 4
n, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
W = np.exp(-2j * np.pi * n * k / N)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
im = axes[0].imshow(np.real(W), cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_title("Ma tran twiddle W (phan thuc)")
for i in range(N):
    for j in range(N):
        axes[0].text(j, i, f"{np.real(W[i,j]):.1f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=axes[0], fraction=0.046)
x = np.array([1, 0, 1, 0], dtype=float)
X = W @ x
axes[1].bar(np.arange(N), np.abs(X), color="steelblue", edgecolor="navy")
axes[1].set_title("X = W @ x  (DFT 1D)")
axes[1].set_xlabel("Chi so tan so k")
fig.suptitle("DFT 1D: W[n,k] = exp(-j2pi nk/N)", fontsize=12, fontweight="bold")
save(fig, "03-dft1-twiddle.png")

# 04 - DFT2 hang cot
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")
steps = [
    (0.05, "Anh MxN", "#FFF2CC"),
    (0.28, "DFT tung HANG\n-> temp", "#D5E8D4"),
    (0.51, "DFT tung COT\n-> F(u,v)", "#DAE8FC"),
    (0.74, "Pha/bien do", "#F8CECC"),
]
for x, t, c in steps:
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.45), 0.18, 0.35, boxstyle="round", fc=c, ec="black"))
    ax.text(x + 0.09, 0.62, t, ha="center", va="center", fontsize=10, fontweight="bold")
for x1, x2 in [(0.23, 0.28), (0.46, 0.51), (0.69, 0.74)]:
    ax.annotate("", xy=(x2, 0.62), xytext=(x1, 0.62), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(0.5, 0.15, "DFT 2D = DFT 1D theo hang, roi DFT 1D theo cot (tach bien)", ha="center", fontsize=11)
ax.set_title("DFT 2D — hang roi cot", fontsize=13, fontweight="bold")
save(fig, "04-dft2-hang-cot.png")

# 05 - IDFT
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")
ax.text(0.5, 0.82, "IDFT: conj(DFT(conj(F))) / MN  -> lay phan thuc", ha="center", fontsize=12, fontweight="bold")
for x, t in [(0.1, "F phuc"), (0.4, "conj(F)"), (0.65, "f khoi phuc")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.35), 0.2, 0.3, boxstyle="round", fc="#E1D5E7", ec="black"))
    ax.text(x + 0.1, 0.5, t, ha="center", va="center", fontsize=11)
for x1, x2 in [(0.30, 0.40), (0.60, 0.65)]:
    ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
save(fig, "05-idft.png")

# 06 - loc 4 buoc
fig, ax = plt.subplots(figsize=(11, 3))
ax.axis("off")
labels = ["fft2", "fftshift", "nhan H", "ifftshift", "ifft2", "real"]
xs = np.linspace(0.04, 0.88, len(labels))
for i, (x, lb) in enumerate(zip(xs, labels)):
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.4), 0.12, 0.28, boxstyle="round", fc="#BDD7EE", ec="black"))
    ax.text(x + 0.06, 0.54, lb, ha="center", va="center", fontsize=9, fontweight="bold")
    if i < len(labels) - 1:
        ax.annotate("", xy=(xs[i + 1], 0.54), xytext=(x + 0.12, 0.54), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_title("Quy trinh loc tan so (4 buoc chinh)", fontsize=12, fontweight="bold")
save(fig, "06-loc-4-buoc.png")

# 07 - ideal LP HP
M = N = 64
u = np.arange(M) - M // 2
v = np.arange(N) - N // 2
UU, VV = np.meshgrid(u, v, indexing="ij")
D = np.sqrt(UU ** 2 + VV ** 2)
D0 = 15
H_lp = (D <= D0).astype(float)
H_hp = 1 - H_lp
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(H_lp, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Ideal LP (D0={D0})")
axes[1].imshow(H_hp, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Ideal HP (D0={D0})")
for a in axes:
    a.axis("off")
fig.suptitle("Loc ly tuong: vong tron tan so quanh DC", fontsize=12)
save(fig, "07-ideal-lp-hp.png")

# 08 - Gibbs
x = np.linspace(-3, 3, 400)
ideal = np.sinc(x * 2) * np.exp(-x ** 2 * 0.5)
ringing = ideal + 0.15 * np.sin(8 * x) * np.exp(-0.3 * x ** 2)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x, ideal, "b-", lw=2, label="Ly tuong (mem)")
ax.plot(x, ringing, "r--", lw=2, label="Co Gibbs (vong)")
ax.axhline(0, color="gray", lw=0.5)
ax.legend()
ax.set_title("Hien tuong Gibbs: loc cat cung -> vong (ringing) quanh bien", fontsize=11)
ax.set_xlabel("Vi tri")
save(fig, "08-gibbs.png")

# 09 - Butterworth
D0, n = 20, 2
orders = [1, 2, 5, 20]
fig, ax = plt.subplots(figsize=(8, 4))
r = np.linspace(0, 50, 200)
for order in orders:
    H = 1 / (1 + (r / D0) ** (2 * order))
    ax.plot(r, H, lw=2, label=f"n={order}")
ax.axhline(0.5, color="gray", ls="--", label="H=0.5 tai D0")
ax.axvline(D0, color="gray", ls=":")
ax.set_xlabel("D(u,v)"); ax.set_ylabel("H")
ax.set_title("Butterworth LP: n lon -> gan Ideal", fontsize=11)
ax.legend(); ax.grid(True, alpha=0.3)
save(fig, "09-butterworth.png")

# 10 - bandpass
H_lo = 1 / (1 + (D / 10) ** 4)
H_hi = 1 / (1 + (D / 30) ** 4)
H_bp = H_hi - H_lo
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(H_lo, cmap="viridis"); axes[0].set_title("LP D_high=30"); axes[0].axis("off")
axes[1].imshow(H_hi, cmap="viridis"); axes[1].set_title("LP D_low=10"); axes[1].axis("off")
axes[2].imshow(H_bp, cmap="viridis"); axes[2].set_title("BP = hieu hai LP"); axes[2].axis("off")
fig.suptitle("Band-pass: giu dai tan so [D_low, D_high]", fontsize=12)
save(fig, "10-bandpass.png")

# 11 - notch
H_notch = np.ones_like(D)
for u0, v0 in [(12, 0), (-12, 0)]:
    Dn = np.sqrt((UU - u0) ** 2 + (VV - v0) ** 2)
    H_notch *= 1 / (1 + (3 / (Dn + 0.1)) ** 4)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(H_notch, cmap="gray", vmin=0, vmax=1)
ax.set_title("Notch: chan tan so (u0,v0) + doi xung\n(loai nhieu tuan hoan)", fontsize=10)
ax.axis("off")
save(fig, "11-notch.png")

# 12 - Sobel
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
im = axes[0].imshow(Kx, cmap="RdBu_r", vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, str(Kx[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
axes[0].set_title("Kernel Sobel Kx"); axes[0].axis("off")
plt.colorbar(im, ax=axes[0], fraction=0.046)
demo = np.zeros((7, 7))
demo[2:5, 2:5] = 1
axes[1].imshow(demo, cmap="gray")
axes[1].arrow(3, 1, 0, 4, head_width=0.3, head_length=0.4, fc="red", ec="red", lw=2)
axes[1].set_title("Gradient Ix: do thay doi theo x")
axes[1].axis("off")
fig.suptitle("Sobel tinh dao ham anh", fontsize=12)
save(fig, "12-sobel.png")

# 13 - Harris R
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
ax.text(0.5, 0.9, "Harris: R = det(M) - k*tr(M)^2", ha="center", fontsize=13, fontweight="bold")
for x, t, c in [(0.08, "R > 0\nGOC", "#C6EFCE"), (0.38, "R ~ 0\nPHANG", "#FFEB9C"), (0.68, "R < 0\nCANH", "#FFC7CE")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.35), 0.22, 0.35, boxstyle="round", fc=c, ec="black"))
    ax.text(x + 0.11, 0.52, t, ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(0.5, 0.12, "M = Gaussian(Ix^2, Iy^2, IxIy)", ha="center", fontsize=10)
save(fig, "13-harris-R.png")

# 14 - Harris-Laplace
fig, ax = plt.subplots(figsize=(9, 4))
scales = [1.0, 1.6, 2.5, 4.0]
for i, s in enumerate(scales):
    ax.add_patch(Circle((1.5 + i * 2, 2.5), s * 0.35, fill=False, ec="navy", lw=2))
    ax.text(1.5 + i * 2, 0.8, f"sigma={s}", ha="center", fontsize=9)
ax.plot([0.5, 8.5], [2.5, 2.5], "k--", alpha=0.3)
ax.annotate("Chon sigma* max |LoG|", xy=(5.5, 3.5), fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=1.5), xytext=(3, 4.2))
ax.set_xlim(0, 9); ax.set_ylim(0, 5)
ax.set_title("Harris-Laplace: goc + chon thang do phu hop", fontsize=12)
ax.axis("off")
save(fig, "14-harris-laplace.png")

# 15 - FAST 12 diem
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect("equal")
circle_pts = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
              (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]
for i, (dy, dx) in enumerate(circle_pts):
    c = "limegreen" if i in [0, 4, 8, 12] else "steelblue"
    ax.plot(dx, dy, "o", color=c, ms=8)
    ax.text(dx * 1.15, dy * 1.15, str(i + 1), fontsize=7, ha="center")
ax.plot(0, 0, "r*", ms=20, label="p (tam)")
ax.set_title("FAST: 16 diem vong, can 12 lien tiep\n4 diem xanh = kiem tra nhanh", fontsize=10)
ax.legend(loc="upper right")
ax.axis("off")
save(fig, "15-fast-12-diem.png")

# 16 - so sanh 3 pp
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
rows = [
    ("Harris", "Cham, chinh xac", "Khong bat bien thang do"),
    ("Harris-Laplace", "Bat bien thang do", "Cham hon FAST"),
    ("FAST", "Rat nhanh", "It mo ta (chi vi tri)"),
]
for i, (name, pro, con) in enumerate(rows):
    y = 0.75 - i * 0.28
    ax.text(0.05, y, name, fontsize=12, fontweight="bold")
    ax.text(0.25, y, pro, fontsize=10, color="green")
    ax.text(0.55, y, con, fontsize=10, color="darkred")
ax.set_title("So sanh Harris / Harris-Laplace / FAST", fontsize=13, fontweight="bold")
save(fig, "16-so-sanh-3pp.png")

# 17 - DoG pyramid
fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for i in range(4):
    g = np.random.randn(32 >> (i // 4), 32).cumsum(axis=1)
    axes[0, i].imshow(g, cmap="gray"); axes[0, i].set_title(f"G sigma{i+1}", fontsize=8); axes[0, i].axis("off")
    d = np.random.randn(32, 32) * 0.3
    axes[1, i].imshow(d, cmap="RdBu_r"); axes[1, i].set_title(f"DoG {i}", fontsize=8); axes[1, i].axis("off")
fig.suptitle("Kim tu thap Gaussian + DoG (octave)", fontsize=12)
save(fig, "17-dog-pyramid.png")

# 18 - extrema 26
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
# fallback 2d if no 3d
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    layers = [np.random.rand(5, 5) for _ in range(3)]
    cx, cy = 2, 2
    for z, L in enumerate(layers):
        for i in range(5):
            for j in range(5):
                c = "red" if z == 1 and i == cx and j == cy else "lightgray"
                ax.bar3d(j, i, z, 0.8, 0.8, L[i, j] * 0.3, color=c, alpha=0.8)
    ax.set_title("Cuc tri 26 lang gieng (3 tang DoG)")
except Exception:
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = np.ones((3, 3))
    ax.imshow(grid, cmap="Blues", alpha=0.3)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                ax.add_patch(Circle((1, 1), 0.15, color="red"))
                ax.text(1, 1, "*", ha="center", va="center", color="white", fontweight="bold")
            else:
                ax.plot(1 + dj * 0.4, 1 + di * 0.4, "ko", ms=6)
    ax.text(0.5, -0.08, "Tang tren + giua + duoi: tong 26 lang gieng", transform=ax.transAxes, ha="center")
    ax.set_title("Cuc tri 26 lang gieng trong DoG")
    ax.axis("off")
save(fig, "18-extrema-26.png")

# 19 - huong
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 6); ax.set_ylim(-1, 6)
ax.add_patch(Circle((2.5, 2.5), 2, fill=False, ec="gray", ls="--"))
angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
hist = np.abs(np.sin(3 * angles)) + 0.2
for i, (a, h) in enumerate(zip(angles, hist)):
    r1, r2 = 2.2, 2.2 + h * 1.5
    ax.plot([2.5 + r1 * np.cos(a), 2.5 + r2 * np.cos(a)],
            [2.5 + r1 * np.sin(a), 2.5 + r2 * np.sin(a)], color="steelblue", lw=1.5)
best = int(np.argmax(hist))
a = angles[best]
ax.arrow(2.5, 2.5, 2.8 * np.cos(a), 2.8 * np.sin(a), head_width=0.2, fc="red", ec="red", lw=2)
ax.set_title("Histogram 36 bin -> huong chu dao theta*", fontsize=11)
ax.axis("off")
save(fig, "19-huong.png")

# 20 - descriptor
fig, ax = plt.subplots(figsize=(5, 5))
for i in range(2):
    for j in range(2):
        ax.add_patch(mpatches.Rectangle((j * 2, 1 - i * 2), 2, 2, fill=False, ec="black", lw=2))
        ax.text(j * 2 + 1, 1 - i * 2 + 1, f"O({i},{j})\n8 bin", ha="center", va="center", fontsize=9)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-3.5, 2.5)
ax.set_title("Descriptor 32D = 2x2 o x 8 bin huong\n(trong he toa do xoay theo theta*)", fontsize=10)
ax.axis("off")
save(fig, "20-descriptor.png")

# 21 - phan cong nhom (also copy to OUT)
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
members = [
    ("Duy", "DFT §1.1", "#5B9BD5", "Done"),
    ("Tuyen", "apply_filter, ideal §1.3-1.4", "#ED7D31", "TODO"),
    ("Truong", "Butterworth, notch §1.5-1.7", "#70AD47", "TODO"),
    ("Nguyen", "Sobel, Harris §2.0-2.1", "#FFC000", "TODO"),
    ("Nhung", "H-Laplace, FAST §2.2-2.4", "#7030A0", "TODO"),
    ("Quyen", "SIFT §3.1-3.5", "#C00000", "TODO"),
]
for i, (name, task, color, st) in enumerate(members):
    y = 0.82 - i * 0.14
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y - 0.05), 0.9, 0.1, boxstyle="round", fc=color, ec="black", alpha=0.85))
    ax.text(0.08, y, name, fontsize=11, fontweight="bold", color="white", va="center")
    ax.text(0.22, y, task, fontsize=9, color="white", va="center")
    ax.text(0.88, y, st, fontsize=9, color="white", va="center", ha="right", fontweight="bold")
ax.set_title("Nhom C — Phan cong Chapter 2", fontsize=14, fontweight="bold")
save(fig, "21-phan-cong-nhom.png")

# Gantt deadline
fig, ax = plt.subplots(figsize=(11, 4))
tasks = [
    ("Duy: DFT + phan chia + hinh", 0, 1, "#5B9BD5"),
    ("Tuyen: loc ly tuong", 0.5, 1.5, "#ED7D31"),
    ("Truong: BW + notch", 0.5, 1.5, "#70AD47"),
    ("Nguyen: Sobel + Harris", 0.5, 1.5, "#FFC000"),
    ("Nhung: H-Laplace + FAST", 0.5, 1.5, "#7030A0"),
    ("Quyen: SIFT", 0.5, 1.5, "#C00000"),
    ("Chay full notebook + nop", 1, 2, "#A5A5A5"),
]
for i, (name, start, end, color) in enumerate(tasks):
    ax.barh(i, end - start, left=start, height=0.6, color=color, edgecolor="black")
    ax.text(start + 0.05, i, name, va="center", fontsize=8, color="white", fontweight="bold")
ax.axvline(1, color="red", ls="--", lw=2, label="30/5 Duy push")
ax.axvline(2, color="darkred", ls="-", lw=2.5, label="DEADLINE 31/5 23:59")
ax.set_yticks([]); ax.set_xlim(0, 2.1)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_xticklabels(["29/5", "30/5", "30/5 PM", "31/5", "31/5 23:59"])
ax.set_title("Nhom C — Lich Chapter 2", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
fig.savefig(ROOT / "deadline-gantt-nhomE-ch2.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 00 doc
(OUT / "00-DOC-CO-LOAI-GI.txt").write_text(
    """FOLDER giai-thich-ch2/ — HINH HOC GIAI THICH (KHONG NOP)
============================================================
  01-anh-mau.png        Anh thu nghiem co cau truc
  02-dft-y-nghia.png    DFT: khong gian -> tan so
  03-dft1-twiddle.png   Ma tran W, DFT 1D
  04-dft2-hang-cot.png  DFT 2D tach hang/cot
  05-idft.png           IDFT khoi phuc anh
  06-loc-4-buoc.png     fft2 -> shift -> nhan H -> ifft2
  07-ideal-lp-hp.png    Loc ly tuong thap/cao
  08-gibbs.png          Ringing sau loc cat cung
  09-butterworth.png    Loc mem, bac n
  10-bandpass.png       Giu dai tan so
  11-notch.png          Chan nhieu tuan hoan
  12-sobel.png          Gradient anh
  13-harris-R.png       Bieu do phan hoi R
  14-harris-laplace.png Chon thang do
  15-fast-12-diem.png   Vong 16 diem FAST
  16-so-sanh-3pp.png    Harris vs H-L vs FAST
  17-dog-pyramid.png    Kim tu thap Gaussian/DoG
  18-extrema-26.png     Cuc tri 26 lang gieng
  19-huong.png          Histogram huong SIFT
  20-descriptor.png     Descriptor 32D
  21-phan-cong-nhom.png Ai lam gi

FILE NOP: assignment_starter-2.ipynb (moi thanh vien dien phan minh)
PDF: Computer_Vision 21.pdf
LOCAL (Duy only): _local/DUY-full-reference.ipynb — KHONG PUSH
""",
    encoding="utf-8",
)

print("OK", len(list(OUT.glob("*.png"))), "PNG + gantt")
