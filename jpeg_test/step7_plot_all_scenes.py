import os
import sys
import subprocess
import matplotlib.pyplot as plt

def get_abs_path(rel_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, rel_path)

def main():
    scenes = [
        "courtyard", "delivery_area", "electro", 
        "kicker", "meadow", "office", 
        "pipes", "playground", "relief"
    ]
    
    script5 = get_abs_path("jpeg_test/step5_eval_full.py")
    
    # Store metrics for each scene
    all_data = {
        "qualities": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "trans": {},
        "rot": {},
        "chamfer": {}
    }
    
    python_exe = sys.executable
    
    for scene in scenes:
        print(f"Parsing metrics for {scene}...")
        res = subprocess.run(f"{python_exe} {script5} {scene}", shell=True, capture_output=True, text=True)
        
        scene_trans = []
        scene_rot = []
        scene_chamfer = []
        
        lines = res.stdout.split('\n')
        for line in lines:
            if line.startswith("Q="):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 10:
                    try:
                        phys_mean_cm = float(parts[3])
                        r_mean = float(parts[5])
                        chamfer = float(parts[9])
                        scene_trans.append(phys_mean_cm)
                        scene_rot.append(r_mean)
                        scene_chamfer.append(chamfer)
                    except ValueError:
                        pass
        
        if len(scene_trans) == 9:
            all_data["trans"][scene] = scene_trans
            all_data["rot"][scene] = scene_rot
            all_data["chamfer"][scene] = scene_chamfer
        else:
            print(f"[Warning] Failed to completely parse 9 qualities for {scene}")
    
    # Now plot everything
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('VGGT Multi-Scene Stability Evaluation against JPEG Compression', fontsize=20, fontweight='bold', y=1.05)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = plt.cm.tab10(range(9))
    
    # Plot 1: Translation Mean
    ax1 = axes[0]
    for i, scene in enumerate(scenes):
        if scene in all_data["trans"]:
            ax1.plot(all_data["qualities"], all_data["trans"][scene], marker=markers[i], color=colors[i], linewidth=2, label=scene)
    ax1.set_title('1. Camera Translation Drift (cm)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('JPEG Quality (Q-value)', fontsize=12)
    ax1.set_ylabel('Mean Error (cm)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 2: Rotation Error (Mean)
    ax2 = axes[1]
    for i, scene in enumerate(scenes):
        if scene in all_data["rot"]:
            ax2.plot(all_data["qualities"], all_data["rot"][scene], marker=markers[i], color=colors[i], linewidth=2, label=scene)
    ax2.set_title('2. Camera Rotation Error (Degree)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('JPEG Quality (Q-value)', fontsize=12)
    ax2.set_ylabel('Mean Angle Error (Degree)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 3: Overall Chamfer Distance
    ax3 = axes[2]
    for i, scene in enumerate(scenes):
        if scene in all_data["chamfer"]:
            ax3.plot(all_data["qualities"], all_data["chamfer"][scene], marker=markers[i], color=colors[i], linewidth=2, label=scene)
    ax3.set_title('3. 3D Pointmap Deformation (Chamfer, cm)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('JPEG Quality (Q-value)', fontsize=12)
    ax3.set_ylabel('Overall Reconstruction Error (cm)', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    vis_dir = get_abs_path("jpeg_test/vis_plot")
    os.makedirs(vis_dir, exist_ok=True)
    plot_file = os.path.join(vis_dir, "VGGT_All_9_Scenes_Comparison.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Master plot generated with all scenes: {plot_file}")

if __name__ == "__main__":
    main()
