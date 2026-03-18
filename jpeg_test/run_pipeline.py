import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_bash(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def get_abs_path(rel_path):
    # 获取当前脚本所在目录的最外层（即vggt_test目录）并向内拼接
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, rel_path)

def run_pipeline(scene):
    print(f"\n" + "="*50)
    print(f"🚀 STARTING PIPELINE FOR SCENE: {scene}")
    print("="*50 + "\n")
    
    script1 = get_abs_path("jpeg_test/step1_compress_eth3d.py")
    script2 = get_abs_path("jpeg_test/step2_run_vggt.py")
    script5 = get_abs_path("jpeg_test/step5_eval_full.py")
    
    # Run Step 1
    run_bash(f"python {script1} {scene}")
    
    # Run Step 2
    run_bash(f"python {script2} {scene}")
    
    # Run Step 5 and capture output
    res = subprocess.run(f"python {script5} {scene}", shell=True, capture_output=True, text=True)
    step5_out = res.stdout
    print(step5_out)
    
    log_file = get_abs_path("jpeg_test/test_log.txt")
    with open(log_file, "a") as f:
        f.write(f"\n\n{'='*50}\nSCENE: {scene}\n{'='*50}\n")
        f.write(step5_out)
    
    # Parse metrics for plotting
    qualities = []
    mean_trans, max_trans = [], []
    mean_rot, max_rot = [], []
    overall_chamfer = []
    
    for line in step5_out.split('\n'):
        if line.startswith("Q="):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 10:
                q_val = int(parts[0].replace("Q=", "").strip())
                phys_mean_cm = float(parts[3])
                phys_max_cm = float(parts[4])
                r_mean = float(parts[5])
                r_max = float(parts[6])
                chamfer = float(parts[9])
                
                qualities.append(q_val)
                mean_trans.append(phys_mean_cm)
                max_trans.append(phys_max_cm)
                mean_rot.append(r_mean)
                max_rot.append(r_max)
                overall_chamfer.append(chamfer)
                
    if not qualities:
        print(f"Failed to parse metrics for {scene}")
        return
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f'VGGT Profiling against JPEG Compression (Scene: {scene})', fontsize=18, fontweight='bold', y=1.05)
    
    ax1 = axes[0]
    ax1.plot(qualities, max_trans, marker='s', color='#d62728', linewidth=2.5, label='Max Drift')
    ax1.plot(qualities, mean_trans, marker='o', color='#1f77b4', linewidth=2.5, label='Mean Drift')
    ax1.set_title('1. Camera Translation Drift (cm)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('JPEG Quality', fontsize=12)
    ax1.set_ylabel('Error (cm)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    ax2 = axes[1]
    ax2.plot(qualities, max_rot, marker='s', color='#ff7f0e', linewidth=2.5, label='Max Rotation Error')
    ax2.plot(qualities, mean_rot, marker='o', color='#2ca02c', linewidth=2.5, label='Mean Rotation Error')
    ax2.set_title('2. Camera Rotation Error (Degree)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('JPEG Quality', fontsize=12)
    ax2.set_ylabel('Error (Degree)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    ax3 = axes[2]
    ax3.plot(qualities, overall_chamfer, marker='D', color='#9467bd', linewidth=3, label='Overall Chamfer')
    ax3.set_title('3. 3D Pointmap Deformation (Chamfer, cm)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('JPEG Quality', fontsize=12)
    ax3.set_ylabel('Error (cm)', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    vis_dir = get_abs_path("jpeg_test/vis_plot")
    os.makedirs(vis_dir, exist_ok=True)
    plot_file = os.path.join(vis_dir, f"VGGT_Compression_Metrics_{scene}.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to: {plot_file}")

if __name__ == "__main__":
    scenes = [
        "kicker",
        "meadow",
        "office",
        "relief",
    ]
    log_file = get_abs_path("jpeg_test/test_log.txt")
    # Clear log
    with open(log_file, "w") as f:
        f.write("VGGT JPEG COMPRESSION EXPERIMENT LOG FOR REMAINING SCENES\n")
        
    for scene in scenes:
        run_pipeline(scene)
