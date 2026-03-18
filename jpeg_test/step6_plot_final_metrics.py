import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. 严格核对并录入你 test_log.txt 中的 9 级梯度数据
    # 横坐标：图像质量 Q (越小代表压缩越严重，画质越差)
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # 维度一：物理平移漂移 (单位：厘米 cm)
    mean_trans = [9.91,  7.02,  2.52,  1.19,  1.56,  1.23,  1.45,  0.81,  0.45]
    max_trans  = [19.74, 17.87, 8.28,  2.47,  3.62,  2.26,  3.52,  2.20,  1.42]
    
    # 维度二：旋转角度误差 (单位：度 Degree)
    mean_rot   = [0.4830, 0.3414, 0.2482, 0.1511, 0.1308, 0.1098, 0.1501, 0.1013, 0.0764]
    max_rot    = [0.8900, 1.0091, 0.7982, 0.3795, 0.3877, 0.2149, 0.4511, 0.3226, 0.2017]
    
    # 维度三：3D点云综合倒角距离 Overall Chamfer Distance (单位：厘米 cm)
    overall_chamfer = [11.67, 8.07, 3.59, 2.55, 2.83, 2.48, 2.41, 2.39, 2.29]

    # 2. 初始化全局画布 (1行3列的宽幅图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('VGGT Robustness Profiling against JPEG Compression (ETH3D "pipes" Scene)', 
                 fontsize=18, fontweight='bold', y=1.05)

    # ================= 子图 1: 物理平移漂移 =================
    ax1 = axes[0]
    ax1.plot(qualities, max_trans, marker='s', color='#d62728', linewidth=2.5, markersize=7, label='Max Drift')
    ax1.plot(qualities, mean_trans, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Mean Drift')
    
    # 添加 Q=20 拐点垂直参考线
    ax1.axvline(x=20, color='gray', linestyle='--', alpha=0.7)
    ax1.text(22, 18, 'Crash Point (Q<30)', color='gray', fontsize=11, fontweight='bold')

    ax1.set_title('1. Camera Translation Drift (cm)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('JPEG Quality (Q-value) $\\rightarrow$ Better Quality', fontsize=12)
    ax1.set_ylabel('Absolute Physical Error (cm)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xticks(qualities)

    # ================= 子图 2: 旋转角度误差 =================
    ax2 = axes[1]
    ax2.plot(qualities, max_rot, marker='s', color='#ff7f0e', linewidth=2.5, markersize=7, label='Max Rotation Error')
    ax2.plot(qualities, mean_rot, marker='o', color='#2ca02c', linewidth=2.5, markersize=7, label='Mean Rotation Error')
    
    ax2.axvline(x=20, color='gray', linestyle='--', alpha=0.7)

    ax2.set_title('2. Camera Rotation Error (Degree)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('JPEG Quality (Q-value) $\\rightarrow$ Better Quality', fontsize=12)
    ax2.set_ylabel('Angle Error (Degrees)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_xticks(qualities)

    # ================= 子图 3: 3D点云倒角距离 =================
    ax3 = axes[2]
    ax3.plot(qualities, overall_chamfer, marker='D', color='#9467bd', linewidth=3, markersize=8, label='Overall Chamfer')
    
    # 填充安全区 (Q>=40) 的背景颜色，直观显示模型抗性
    ax3.axvspan(40, 90, facecolor='#e6f2ff', alpha=0.5, label='Robustness Zone')
    ax3.axvline(x=20, color='gray', linestyle='--', alpha=0.7)

    ax3.set_title('3. 3D Pointmap Deformation (Chamfer, cm)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('JPEG Quality (Q-value) $\\rightarrow$ Better Quality', fontsize=12)
    ax3.set_ylabel('Reconstruction Error (cm)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.set_xticks(qualities)

    # 3. 调整布局并保存
    plt.tight_layout()
    save_path = "VGGT_Compression_Tradeoff_Full_Metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 学术级 1x3 组图已生成并保存至: {save_path}")

if __name__ == "__main__":
    main()