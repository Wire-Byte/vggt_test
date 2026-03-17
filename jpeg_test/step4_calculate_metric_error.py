import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R

def get_real_camera_centers(images_txt_path):
    """
    解析 COLMAP 的 images.txt，获取每张图片真实的物理相机中心坐标（单位：米）
    """
    cam_centers = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('#'): continue
        parts = line.strip().split()
        # COLMAP images.txt 的位姿行包含至少 10 个元素
        if len(parts) >= 10 and parts[8].isdigit():
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            filename = os.path.basename(parts[9])
            
            # COLMAP 的 [R|T] 是 World-to-Camera (W2C)
            # 真实的相机中心在世界坐标系下是: C = -R^T * T
            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix() # scipy 顺序是 x,y,z,w
            t_vec = np.array([tx, ty, tz])
            center = -np.dot(rot_matrix.T, t_vec)
            
            cam_centers[filename] = center
            
    return cam_centers

def get_vggt_camera_centers(extrinsics_npy):
    """
    从 VGGT 输出的 Nx3x4 外参矩阵中提取相机中心
    """
    extrinsics = np.load(extrinsics_npy) # Shape: [N, 3, 4]
    centers = []
    for ext in extrinsics:
        rot_matrix = ext[:3, :3]
        t_vec = ext[:3, 3]
        # 同样，计算 C = -R^T * T
        center = -np.dot(rot_matrix.T, t_vec)
        centers.append(center)
    return np.array(centers)

def main():
    scene_name = "pipes"
    
    # 1. 路径配置 (请确保路径正确)
    images_txt_path = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/multi_view_training_dslr_undistorted/{scene_name}/dslr_calibration_undistorted/images.txt"
    vggt_results_dir = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_results"
    image_folder = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_compressed/Q_100"
    
    # 2. 获取真实物理相机坐标
    real_cam_dict = get_real_camera_centers(images_txt_path)
    
    # 保证顺序和 VGGT 推理时的排序一模一样 (非常重要)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.JPG")))
    image_names = [os.path.basename(p) for p in image_paths]
    
    real_centers = np.array([real_cam_dict[name] for name in image_names])
    
    # 3. 获取 VGGT 的归一化相机坐标 (使用 Q=100 作为基准)
    vggt_poses_path = os.path.join(vggt_results_dir, "camera_poses_Q100.npy")
    vggt_centers = get_vggt_camera_centers(vggt_poses_path)
    
    # 4. 计算缩放系数 S (Scale Factor)
    # 通过计算所有相机对之间的平均距离来求比例，这样最稳健
    real_dists = []
    vggt_dists = []
    num_cams = len(real_centers)
    for i in range(num_cams):
        for j in range(i+1, num_cams):
            real_dists.append(np.linalg.norm(real_centers[i] - real_centers[j]))
            vggt_dists.append(np.linalg.norm(vggt_centers[i] - vggt_centers[j]))
            
    mean_real_dist = np.mean(real_dists)
    mean_vggt_dist = np.mean(vggt_dists)
    
    scale_factor = mean_real_dist / mean_vggt_dist
    
    print("========== 尺度对齐分析 (Scale Alignment) ==========")
    print(f"场景中相机间的平均物理距离: {mean_real_dist:.3f} 米")
    print(f"VGGT 空间中相机间的平均距离: {mean_vggt_dist:.3f} 单位")
    print(f"-> 测算出的缩放系数 S = {scale_factor:.3f} (即 VGGT 的 1.0 等于 {scale_factor:.3f} 米)")
    
    # 5. 将你的相对误差转化为物理误差
    # 填入你上一轮跑出来的数据
    q10_mean_error_unitless = 0.03163
    q10_max_error_unitless = 0.06303
    
    q10_mean_error_meters = q10_mean_error_unitless * scale_factor
    q10_max_error_meters = q10_max_error_unitless * scale_factor
    
    print("\n========== 最终科学报告指标 (Physical Error) ==========")
    print(f"在极度压缩 (Q=10) 下，VGGT 的预测漂移换算为真实物理空间：")
    print(f"平均漂移量: {q10_mean_error_meters:.4f} 米 (即 {q10_mean_error_meters * 100:.2f} 厘米)")
    print(f"最大单帧漂移: {q10_max_error_meters:.4f} 米 (即 {q10_max_error_meters * 100:.2f} 厘米)")

if __name__ == "__main__":
    main()