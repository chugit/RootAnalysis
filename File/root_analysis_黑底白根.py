import cv2
import numpy as np
from skimage import morphology
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from natsort import natsorted

# 设置工作目录
work_dir = 'D:\\R\\RootAnalysis' # 在该目录下自行创建Input文件夹
input_dir = os.path.join(work_dir, 'Input') # 原始图像放于Input文件夹
output_dir = os.path.join(work_dir, 'Output') # 定义二值化图像、根骨架图像、结果数据的导出位置
output_csv = os.path.join(output_dir, 'root_analysis_results.csv') # 定义结果数据的导出文件名
os.makedirs(output_dir, exist_ok=True)

# 定义像素到厘米的转换比例
pixels_per_cm = 130  # 130像素对应1cm

# 定义图像裁剪和阈值方法的相关参数
crop_top_percentage = 0.0985  # 裁除图像顶部的x%。0为不剪裁
crop_bottom_percentage = 0.00  # 裁除图像底部的x%。0为不剪裁
crop_left_percentage = 0.00  # 裁除图像左部的x%。0为不剪裁
crop_right_percentage = 0.00  # 裁除图像右部的x%。0为不剪裁
threshold_method = "fixed"  # 阈值方法选择："fixed" 或 "otsu"（固定阈值法或Otsu自动法）
fixed_threshold_value = 30 # 15.30.20.45.指定固定阈值（仅在使用固定阈值法时有效），阈值越小，根系越厚

########## 下方代码全选运行，无需调整 #############################

# 定义图像处理函数
def process_image(image_file):
    # 构建图像路径并加载图像
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查图像是否加载成功
    if image is None:
        print(f"无法加载图像 {image_file}，跳过")
        return None

    # 裁剪图像，以移除干扰物所在区域，避免干扰分析
    height, width = image.shape
    crop_top = int(height * crop_top_percentage)
    crop_bottom = int(height * crop_bottom_percentage)
    crop_left = int(width * crop_left_percentage)
    crop_right = int(width * crop_right_percentage)
    image_cropped = image[crop_top:height - crop_bottom, crop_left:width - crop_right]
    
    # 根据选择的阈值方法生成二值化图像
    if threshold_method == "fixed":
        _, binary = cv2.threshold(image_cropped, fixed_threshold_value, 255, cv2.THRESH_BINARY)
        threshold_value = fixed_threshold_value
    elif threshold_method == "otsu":
        threshold_value, binary = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        print(f"未定义的阈值方法: {threshold_method}")
        return None
    
    # （可选）通过腐蚀进一步减小根的边界厚度
    # kernel = np.ones((2, 2), np.uint8)
    # binary = cv2.erode(binary, kernel, iterations=1)
    
    # 提取根骨架
    skeleton = morphology.skeletonize(binary // 255).astype(np.uint8) * 255

    # 导出处理后的图像
    base_name = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_binary.jpg'), binary) # 二值化图像
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_skeleton.jpg'), skeleton) # 根骨架图像

    # 计算总根长
    pixel_length = np.sum(skeleton) / 255
    total_root_length_cm = pixel_length / pixels_per_cm

    # 计算根系投影面积
    projected_area = np.sum(binary // 255) / (pixels_per_cm**2)

    # 计算根系平均直径
    average_root_diameter_mm = (projected_area / total_root_length_cm) * 10

    # 计算根系总表面积和总体积
    root_surface_area = np.pi * (average_root_diameter_mm / 10) * total_root_length_cm
    root_volume = np.pi * ((average_root_diameter_mm / 2) / 10)**2 * total_root_length_cm

    # 返回结果
    return {
        "Image": image_file,
        "Threshold Value": threshold_value,
        "Projected Area (cm2)": projected_area,
        "Total Root Length (cm)": total_root_length_cm,
        "Average Root Diameter (mm)": average_root_diameter_mm,
        "Root Surface Area (cm2)": root_surface_area,
        "Root Volume (cm3)": root_volume
    }

# 初始化结果列表
results = []

# 多线程并行运算
start_time = time.time()
with ThreadPoolExecutor(max_workers=6) as executor: # 此处调用的CPU线程数限定为6
    futures = {executor.submit(process_image, image_file): image_file for image_file in os.listdir(input_dir)}
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            results.append(result)
end_time = time.time()
print(f"多线程并行运算时间: {end_time - start_time:.2f} s")

# 结果按文件名的自然顺序排序
results = natsorted(results, key=lambda x: x["Image"])

# 导出结果
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

# 打开 Output 文件夹
os.startfile(output_dir)
