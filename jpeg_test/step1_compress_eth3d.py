import os
import glob
from PIL import Image
from tqdm import tqdm

def process_eth3d_scene(scene_input_dir, output_base_dir, qualities=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    # 获取该场景下所有的原图 JPG (根据大小写可以适当调整通配符)
    img_paths = sorted(glob.glob(os.path.join(scene_input_dir, "*.JPG")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(scene_input_dir, "*.jpg")))

    if not img_paths:
        print(f"未在 {scene_input_dir} 找到 JPG/jpg 图片！")
        return

    print(f"找到 {len(img_paths)} 张原图，开始压缩处理...")
    
    bpp_results = {}
    
    for q in qualities:
        # 为不同压缩率创建带有标识的单独子文件夹
        q_dir = os.path.join(output_base_dir, f"Q_{q}")
        os.makedirs(q_dir, exist_ok=True)
        
        total_bpp = 0.0
        
        for img_path in tqdm(img_paths, desc=f"Generating Q={q}"):
            filename = os.path.basename(img_path)
            save_path = os.path.join(q_dir, filename)
            
            # 读取原始图像将其转换 RGB 形式并直接用低 Quality 压缩覆盖保存
            img = Image.open(img_path).convert('RGB')
            # 存为 JPEG 格式进行质量打折，如果 q=100 就保持极高画质
            img.save(save_path, 'JPEG', quality=q)
            
            # 计算客观指标 BPP (Bits Per Pixel)
            file_size_bytes = os.path.getsize(save_path)
            width, height = img.size
            bpp = (file_size_bytes * 8) / (width * height)
            total_bpp += bpp
            
        avg_bpp = total_bpp / len(img_paths)
        bpp_results[q] = avg_bpp
        print(f"-> 平均数据量: {avg_bpp:.4f} bpp 对于质量 Q={q}")

if __name__ == "__main__":
    # 配置为你自己的实际路径，用 pipes 进行测试
    SCENE_NAME = "pipes"
    # 根据你的文件管理器目录适配，下面是针对你 F 盘下载挂载的路径
    INPUT_DIR = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/multi_view_training_dslr_undistorted/{SCENE_NAME}/images/dslr_images_undistorted" 
    
    # 存放在你指定的用来管理实验产出的文件夹（方便我们做版本控制）
    OUTPUT_DIR = f"/NEW_EDS/JJ_Group/shiyc2603/vggt_test/jpeg_test/VGGT_EXP/{SCENE_NAME}_compressed"
    
    process_eth3d_scene(INPUT_DIR, OUTPUT_DIR)