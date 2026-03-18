import os
import sys
import glob
import torch
import numpy as np

# 自动将项目根目录加入环境变量，解决找不到 vggt 和 visual_util 的问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你项目中的核心组件模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from visual_util import predictions_to_glb
import subprocess

def get_freest_gpu():
    """通过 nvidia-smi 动态获取当前空闲显存最多的 GPU"""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # 调用 nvidia-smi 查询所有显卡的剩余显存
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        free_memory = [int(x.strip()) for x in smi_output.strip().split('\n')]
        best_gpu = free_memory.index(max(free_memory))
        print(f"[*] 动态分配最空闲的显卡: cuda:{best_gpu} (空闲显存: {max(free_memory)} MiB)")
        return f"cuda:{best_gpu}"
    except Exception as e:
        print(f"[*] 动态匹配GPU失败，默认降级为 cuda:0")
        return "cuda:0"

def main():
    import sys
    device = get_freest_gpu()
    # 依据你的显卡性质决定浮点数形式
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("正在加载 VGGT 核心模型 ...")
    model = VGGT()
    # 模型全参文件位置在这，如果没有，请按照自己真实位置填写
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
    model = model.to(device)
    model.eval()  # 推理请必须开启 eval 模式

    scene_name = sys.argv[1] if len(sys.argv) > 1 else "pipes"
    # 这里请保持与第一步输出保存的位置相同
    base_data_dir = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_compressed"
    output_glb_dir = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{scene_name}_results"
    
    os.makedirs(output_glb_dir, exist_ok=True)
    
    # 测试的质量阶梯
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        for q in qualities:
            print(f"\n======== 开始处理: Q={q} 的重建 ========")
            glb_filename = os.path.join(output_glb_dir, f"{scene_name}_recon_Q{q}.glb")
            if os.path.exists(glb_filename):
                print(f"Skipping Q={q}, 结果已存在，为了节省时间直接跳过，如需重跑请删掉 GLB 文件")
                continue

            q_dir = os.path.join(base_data_dir, f"Q_{q}")
            
            # 使用项目原生的 load_and_preprocess_images 直接拉进 Batch 张量 (这会确保你的中心剪裁合乎标准)
            image_paths = sorted(glob.glob(os.path.join(q_dir, "*.jpg")) + glob.glob(os.path.join(q_dir, "*.JPG")))
            
            if not image_paths:
                print(f"跳过找不到图的目录：{q_dir}")
                continue
                
            # 项目封装函数会返回标准的 [Batch_N, C, H, W] 并附带做好了张量除以 255 及处理
            # 此处加上 batch 维度让模型识别这是一整个序列视频块: => [1, N, C, H, W]，N
            # 是一个 Batch (1)，里面包含一个由 N 张图组成的序列
            #B=1: 任务批次 (Task Batch)。表示当前正在处理几个“独立的场景重建任务”。这里设为 1，表示一次只重建一个场景。
            #N: 视图数量 (Num Views)。表示同一个场景下，有多少张相互关联的图片。
            images_tensor = load_and_preprocess_images(image_paths, mode="crop").unsqueeze(0).to(device)
            
            print(f"正在送入网络前向推理, 张量维度: {images_tensor.shape} ...")
            outputs = model(images_tensor)

            # --- 下方是对核心数据的重整以及留存输出 ---
            # 从 VGGT 输出的稠密位姿编码解码为标准相机外参 [N, 3, 4] 矩阵
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            H, W = images_tensor.shape[3], images_tensor.shape[4]
            extrinsics, _ = pose_encoding_to_extri_intri(
                outputs["pose_enc"], 
                image_size_hw=(H, W), 
                build_intrinsics=False
            )

            # 为了能够把结果喂给自带的可视化/保存 GLB 功能，我们需要将它们从 GPU 拿出回到 CPU-Numpy 平面
            predictions = {
                # 真实输出变量名叫 world_points, [1, N, H, W, 3] 取出 batch 维度变为 [N, H, W, 3]
                "world_points": outputs["world_points"].detach().cpu().numpy()[0],
                # 如果缺少直接提供 confidence 参数，可利用 depth 的张量框架伪造高置信度 [1, N, H, W] -> [N, H, W]
                "world_points_conf": torch.ones_like(outputs["depth"].squeeze(-1)).detach().cpu().numpy()[0], 
                # [1, N, 3, H, W] 转置回去，变为展示所需 [N, H, W, 3]
                "images": (images_tensor.squeeze(0).permute(0,2,3,1).detach().cpu().numpy() * 255.0).astype(np.uint8),
                # 模型的 Extrinsic 解码后取 [N, 3, 4] 矩阵阵列
                "extrinsic": extrinsics.detach().cpu().numpy()[0],
            }
            
            # 使用项目内置模块抛出一个 .glb 或者点云对象文件供比对
            glbscene = predictions_to_glb(
                predictions=predictions,
                conf_thres=20.0,            # 过滤多余低信心的噪声点
                filter_by_frames="all",
                mask_black_bg=True,         # 清除空白背景影响
                show_cam=True,              # 保留并在空间展示预测的相机位姿
                target_dir=output_glb_dir,  # 保存的目录
                prediction_mode="Predicted Pointmap"
            )
            
            # 自定义存储 GLB 用于本地利用 3D viewer 进行打开浏览对撞
            glb_filename = os.path.join(output_glb_dir, f"{scene_name}_recon_Q{q}.glb")
            if glbscene is not None:
                # trimesh 会协助我们将整个装配的点云以及相机一同进行打包封存到 .glb 
                glbscene.export(glb_filename)
                print(f"✔ 成功保存直观 3D 文件(包含点云及相机路点): {glb_filename}")
                
            # -- 我们同时也单独存一份原生的相机 Npy 数据供下一步量化分析对比 ---
            np.save(os.path.join(output_glb_dir, f"camera_poses_Q{q}.npy"), predictions["extrinsic"])
            
            # --- 新增：保存点云数据供后续 Chamfer Distance 评测 ---
            # 为了防止文件过大，我们采用 float16 取近似保存，并直接展平
            np.save(os.path.join(output_glb_dir, f"world_points_Q{q}.npy"), predictions["world_points"].astype(np.float16))
            
            print(f"✔ 成功保存相机外参字典及点云矩阵用于定量评测")
            
            # 及时清理内存，防止多个 Quality 循环导致显存累积 OOM
            del outputs
            del predictions
            del extrinsics
            del images_tensor
            del glbscene
            import gc
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()