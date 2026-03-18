import os
import numpy as np

def calculate_translation_error(pose1, pose2):
    """ 计算两个 Nx3x4 或 Nx4x4 位姿矩阵在平移量上的平均和最大误差距离"""
    error_list = []
    # 确保提取的就是针对那同样数目的对应图像帧
    for p1, p2 in zip(pose1, pose2):
        # 简单提取出相机最后一排的推移量矩阵 T 
        t1 = p1[:3, 3] 
        t2 = p2[:3, 3]
        
        # 利用二范数测算两个距离差 
        dist = np.linalg.norm(t1 - t2)
        error_list.append(dist)
    
    return np.mean(error_list), np.max(error_list)

def main():
    import sys
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "pipes"
    output_glb_dir = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_results"
    
    # 我们以最为真实的无损原图所吐出来的结果作为咱们计算其他粗糙版本差距时使用的“基准对比 Ground Truth”
    gt_poses_path = os.path.join(output_glb_dir, "camera_poses_Q100.npy")
    if not os.path.exists(gt_poses_path):
        print(f"找不到高清水准的基础作为锚定 GT : {gt_poses_path}")
        return
        
    gt_poses = np.load(gt_poses_path)
    print(f"基准参考点 (Q=100) 相机位姿已加载。尺寸为: {gt_poses.shape}")
    
    qualities = [10, 30, 50, 70, 90]
    
    print("\n========== 对比分析：压缩下画质降低引发的 VGGT 预测漂移失准评估 ==========")
    for q in qualities:
        test_pose_path = os.path.join(output_glb_dir, f"camera_poses_Q{q}.npy")
        if not os.path.exists(test_pose_path):
            continue
            
        test_poses = np.load(test_pose_path)
        mean_err, max_err = calculate_translation_error(gt_poses, test_poses)
        
        # 此时结合第一步得出的 BPP 客观传输负担大小向你导师去呈现成果将会极具强力学术氛围
        print(f"对比组 [Q={q:<3d}] 与基准原图之相对误差 | 平均平移偏移量: {mean_err:.5f} | 最大离谱单帧丢距: {max_err:.5f}")

if __name__ == "__main__":
    main()