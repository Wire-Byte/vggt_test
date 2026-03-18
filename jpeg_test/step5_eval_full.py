import os
import numpy as np
from scipy.spatial import cKDTree

def get_real_camera_centers(images_txt_path):
    from scipy.spatial.transform import Rotation as R
    cam_centers = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts) >= 10 and parts[8].isdigit():
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            filename = os.path.basename(parts[9])
            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_vec = np.array([tx, ty, tz])
            center = -np.dot(rot_matrix.T, t_vec)
            cam_centers[filename] = center
    return cam_centers

def calculate_pose_errors(pose_gt, pose_test):
    """ 计算两个 Nx3x4 位姿矩阵的平移误差和旋转误差 """
    trans_errors = []
    rot_errors = []
    
    for p_gt, p_test in zip(pose_gt, pose_test):
        # 1. 平移误差 (欧氏距离)
        t_gt, t_test = p_gt[:3, 3], p_test[:3, 3]
        trans_errors.append(np.linalg.norm(t_gt - t_test))
        
        # 2. 旋转误差 (计算两个旋转矩阵的夹角)
        R_gt, R_test = p_gt[:3, :3], p_test[:3, :3]
        # 相对旋转矩阵
        R_rel = np.dot(R_gt, R_test.T)
        trace = np.trace(R_rel)
        # 防止浮点数精度超限
        trace = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(trace))
        rot_errors.append(angle_deg)
        
    return np.mean(trans_errors), np.max(trans_errors), np.mean(rot_errors), np.max(rot_errors)

def calculate_point_cloud_metrics(points_gt, points_test, scale_factor, sample_size=80000):
    """ 
    计算 Accuracy, Completeness 和 Overall Chamfer Distance
    使用 cKDTree 加速，单位将转化为真实物理尺度（厘米）
    """
    # 展平并去除非法 NaN 点
    p_gt = points_gt.reshape(-1, 3)
    p_test = points_test.reshape(-1, 3)
    p_gt = p_gt[~np.isnan(p_gt).any(axis=1)]
    p_test = p_test[~np.isnan(p_test).any(axis=1)]

    # 保证采样数量上限以防 OOM，固定随机种子保证可复现
    np.random.seed(42)
    if len(p_gt) > sample_size:
        p_gt = p_gt[np.random.choice(len(p_gt), sample_size, replace=False)]
    if len(p_test) > sample_size:
        p_test = p_test[np.random.choice(len(p_test), sample_size, replace=False)]
        
    tree_gt = cKDTree(p_gt)
    tree_test = cKDTree(p_test)
    
    # 1. Accuracy: 从重建点云（Test）中找在物理真实（GT）中最近的点
    # 这代表“生成的点云对不对”
    dist_test_to_gt, _ = tree_gt.query(p_test)
    accuracy_m = np.mean(dist_test_to_gt)
    
    # 2. Completeness: 从物理真实（GT）中找在重建点云（Test）里最近的点
    # 这代表“生成的点云全不全”
    dist_gt_to_test, _ = tree_test.query(p_gt)
    completeness_m = np.mean(dist_gt_to_test)
    
    # 3. Overall Chamfer Distance: 综合得分 (均值)
    overall_cd_m = (accuracy_m + completeness_m) / 2.0
    
    # 转成物理厘米 (cm)
    cm_ratio = scale_factor * 100
    return accuracy_m * cm_ratio, completeness_m * cm_ratio, overall_cd_m * cm_ratio

def main():
    import sys
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "pipes"
    output_glb_dir = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_results"
    
    gt_poses_path = os.path.join(output_glb_dir, "camera_poses_Q100.npy")
    gt_points_path = os.path.join(output_glb_dir, "world_points_Q100.npy")
    
    if not os.path.exists(gt_poses_path):
        print(f"找不到基准位姿文件: {gt_poses_path}")
        return
        
    gt_poses = np.load(gt_poses_path)
    
    import glob
    images_txt_path = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/multi_view_training_dslr_undistorted/{scene_name}/dslr_calibration_undistorted/images.txt"
    image_folder = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_compressed/Q_100"
    real_cam_dict = get_real_camera_centers(images_txt_path)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.JPG")))
    image_names = [os.path.basename(p) for p in image_paths]
    real_centers = np.array([real_cam_dict[name] for name in image_names])
    
    vggt_centers = []
    for ext in gt_poses:
        rot_matrix = ext[:3, :3]
        t_vec = ext[:3, 3]
        center = -np.dot(rot_matrix.T, t_vec)
        vggt_centers.append(center)
    vggt_centers = np.array(vggt_centers)
    
    real_dists, vggt_dists = [], []
    num_cams = len(real_centers)
    for i in range(num_cams):
        for j in range(i+1, num_cams):
            real_dists.append(np.linalg.norm(real_centers[i] - real_centers[j]))
            vggt_dists.append(np.linalg.norm(vggt_centers[i] - vggt_centers[j]))
    scale_factor = np.mean(real_dists) / np.mean(vggt_dists)
    print(f"[{scene_name}] Scale Factor calculated as: {scale_factor:.4f}")
    
    calculate_cd = os.path.exists(gt_points_path)
    if calculate_cd:
        gt_points = np.load(gt_points_path)
        print("✔ 发现 Q100 点云文件，准备执行全方位(位姿+几何)质量评测！\n")
    else:
        print("✖ 未发现 Q100 点云！只测相机位姿，不测三维重建几何指标。\n")
    
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    print("=" * 135)
    print(f"{'压缩':<6} | {'相对平移误差 (无单位)':<20} | {'物理平移漂移 (cm)':<20} | {'旋转角度误差 (Degree)':<20} | {'3D点云重建质量分析 / Chamfer Distance (cm)':<45}")
    print(f"{'等级':<6} | {'Mean':<9} | {'Max':<8} | {'Mean':<9} | {'Max':<8} | {'Mean':<9} | {'Max':<8} | {'Accuracy (准)':<14} | {'Completeness (全)':<18} | {'Overall (综合倒角)'}")
    print("-" * 135)
    
    for q in qualities:
        test_pose_path = os.path.join(output_glb_dir, f"camera_poses_Q{q}.npy")
        test_points_path = os.path.join(output_glb_dir, f"world_points_Q{q}.npy")
        
        if not os.path.exists(test_pose_path): continue
            
        test_poses = np.load(test_pose_path)
        rel_mean_t, rel_max_t, mean_r, max_r = calculate_pose_errors(gt_poses, test_poses)
        
        phys_mean_t_cm = rel_mean_t * scale_factor * 100
        phys_max_t_cm = rel_max_t * scale_factor * 100
        
        acc_str, comp_str, overall_str = "N/A", "N/A", "N/A"
        if calculate_cd and os.path.exists(test_points_path):
            test_points = np.load(test_points_path)
            acc, comp, overall = calculate_point_cloud_metrics(gt_points, test_points, scale_factor)
            acc_str, comp_str, overall_str = f"{acc:.2f}", f"{comp:.2f}", f"{overall:.2f}"
            
        print(f"Q={q:<4} | {rel_mean_t:<9.5f} | {rel_max_t:<8.5f} | {phys_mean_t_cm:<9.2f} | {phys_max_t_cm:<8.2f} | {mean_r:<9.4f} | {max_r:<8.4f} | {acc_str:<14} | {comp_str:<18} | {overall_str:<10}")

    print("=" * 135)

if __name__ == "__main__":
    main()
