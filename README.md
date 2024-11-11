本文将介绍植物根系图像中形态参数（总根长、根系平均直径、总表面积、总体积）的批量提取方式，并提供实现批量处理的Python代码。最后，针对图像处理的关键参数，构建具备可交互界面的程序，并将其打包成可执行文件。

# 根系扫描

利用扫描仪、高拍仪、相机等，在固定仪器参数下获取根系图像。同时，在相同条件下获取已知长度的线段/形状的图像，用作校准。

图像中的根系颜色应与背景色有很大的差异，一般分为白底黑根（背景颜色比根系白/亮）或黑底白根（背景颜色比根系黑/深）。

后续分析将在灰度图的基础上进行。

# 根系分析原理

本文使用的根系图像处理流程与WinRHIZO等主流根系分析软件一致，关键步骤包括二值化、骨架化、长度测量、投影面积计算以及通过数学模型计算根系的平均直径、总表面积和总体积。

## 图像二值化

将根系灰度图像通过阈值分割转换为黑白图像，以便区分根系和背景。可以选择固定阈值或Otsu自动阈值法，具体选择基于背景类型和图像光照条件。

二值化后，图像中根系与背景清晰分离，为后续的形态分析提供基础。

## 骨架化处理

通过形态学骨架化将根系区域转换为单像素宽度的线条结构。

骨架化在保持根系主干和分支信息的同时去除了多余的粗度信息，得到的骨架线条能更准确地表征根系的生长路径，为长度测量提供便捷的图像形式。

## 根系长度计算

统计骨架化图像的像素总数，根据像素与实际距离的转换比例（如每厘米的像素数），计算根系总长度。

像素与实际距离的转换比例（本文亦称之为校准系数），可以利用ImageJ、Photoshop等图像处理软件，通过测定已知长度的线段像素点数确定。

## 根系投影面积

统计二值化图像中所有根系像素点总数，根据校准系数，计算根系投影面积。

## 平均根径

将根系投影面积与总长度相除，得到平均根径。

## 根系表面积和总体积

假设根系由多个等直径的小圆柱体组成，表面积和体积可通过根系长度和平均直径计算得来：

$$
\text{总表面积} = \pi \times \text{平均直径} \times \text{总长度}
$$

$$
\text{总体积} = \pi \times \left(\frac{\text{平均直径}}{2}\right)^2 \times \text{总长度}
$$

# 利用Python实现根系图像分析的批量处理

Python及其库/模块的安装方式间[chugit/Crawler_Journal_Abbreviation](https://github.com/chugit/Crawler_Journal_Abbreviation)的[安装Python](https://github.com/chugit/Crawler_Journal_Abbreviation?tab=readme-ov-file#%E5%AE%89%E8%A3%85python/)和[安装Selenium](https://github.com/chugit/Crawler_Journal_Abbreviation?tab=readme-ov-file#%E5%AE%89%E8%A3%85selenium/)部分，其中，将“Selenium”替换为下方代码运行所缺失的库/模块，即为Python库/模块的安装方式。

## 白底黑根

```python
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
input_dir = os.path.join(work_dir, 'Input') # 待处理的原始图像均放于Input文件夹内
output_dir = os.path.join(work_dir, 'Output') # 定义二值化图像、根骨架图像、结果数据的导出位置
output_csv = os.path.join(output_dir, 'root_analysis_results.csv') # 定义结果数据的导出文件名
os.makedirs(output_dir, exist_ok=True)

# 定义像素到厘米的转换比例
pixels_per_cm = 130  # 130像素对应1cm

# 定义图像裁剪和阈值方法的相关参数
crop_top_percentage = 0.001  # 裁除图像顶部的0.1%。0为不剪裁
crop_bottom_percentage = 0  # 裁除图像底部的x%。0为不剪裁
crop_left_percentage = 0  # 裁除图像左部的x%。0为不剪裁
crop_right_percentage = 0  # 裁除图像右部的x%。0为不剪裁
threshold_method = "fixed"  # 阈值方法选择："fixed" 或 "otsu"（固定阈值法或Otsu自动法）
fixed_threshold_value = 70  # 指定固定阈值（仅在使用固定阈值法时有效）。白底黑根，阈值越大，根系越厚

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
        _, binary = cv2.threshold(image_cropped, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
        threshold_value = fixed_threshold_value
    elif threshold_method == "otsu":
        threshold_value, binary = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

```

## 黑底白根

```python
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
input_dir = os.path.join(work_dir, 'Input') # 待处理的原始图像均放于Input文件夹内
output_dir = os.path.join(work_dir, 'Output') # 定义二值化图像、根骨架图像、结果数据的导出位置
output_csv = os.path.join(output_dir, 'root_analysis_results.csv') # 定义结果数据的导出文件名
os.makedirs(output_dir, exist_ok=True)

# 定义像素到厘米的转换比例
pixels_per_cm = 130  # 130像素对应1cm

# 定义图像裁剪和阈值方法的相关参数
crop_top_percentage = 0.001  # 裁除图像顶部的0.1%。0为不剪裁
crop_bottom_percentage = 0  # 裁除图像底部的x%。0为不剪裁
crop_left_percentage = 0  # 裁除图像左部的x%。0为不剪裁
crop_right_percentage = 0  # 裁除图像右部的x%。0为不剪裁
threshold_method = "fixed"  # 阈值方法选择："fixed" 或 "otsu"（固定阈值法或Otsu自动法）
fixed_threshold_value = 15 # 指定固定阈值（仅在使用固定阈值法时有效）。黑底白根，阈值越小，根系越厚

########## 下方代码全选运行，无需调整 #############################

def process_image(image_file):
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"无法加载图像 {image_file}，跳过")
        return None

    height, width = image.shape
    crop_top = int(height * crop_top_percentage)
    crop_bottom = int(height * crop_bottom_percentage)
    crop_left = int(width * crop_left_percentage)
    crop_right = int(width * crop_right_percentage)
    image_cropped = image[crop_top:height - crop_bottom, crop_left:width - crop_right]
    
    if threshold_method == "fixed":
        _, binary = cv2.threshold(image_cropped, fixed_threshold_value, 255, cv2.THRESH_BINARY)
        threshold_value = fixed_threshold_value
    elif threshold_method == "otsu":
        threshold_value, binary = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        print(f"未定义的阈值方法: {threshold_method}")
        return None
    
    skeleton = morphology.skeletonize(binary // 255).astype(np.uint8) * 255

    base_name = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_binary.jpg'), binary)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_skeleton.jpg'), skeleton)

    pixel_length = np.sum(skeleton) / 255
    total_root_length_cm = pixel_length / pixels_per_cm

    projected_area = np.sum(binary // 255) / (pixels_per_cm**2)

    average_root_diameter_mm = (projected_area / total_root_length_cm) * 10

    root_surface_area = np.pi * (average_root_diameter_mm / 10) * total_root_length_cm
    root_volume = np.pi * ((average_root_diameter_mm / 2) / 10)**2 * total_root_length_cm

    return {
        "Image": image_file,
        "Threshold Value": threshold_value,
        "Projected Area (cm2)": projected_area,
        "Total Root Length (cm)": total_root_length_cm,
        "Average Root Diameter (mm)": average_root_diameter_mm,
        "Root Surface Area (cm2)": root_surface_area,
        "Root Volume (cm3)": root_volume
    }

results = []

start_time = time.time()
with ThreadPoolExecutor(max_workers=6) as executor: # 此处调用的CPU线程数限定为6
    futures = {executor.submit(process_image, image_file): image_file for image_file in os.listdir(input_dir)}
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            results.append(result)
end_time = time.time()
print(f"多线程并行运算时间: {end_time - start_time:.2f} s")

results = natsorted(results, key=lambda x: x["Image"])

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

os.startfile(output_dir)

```

# 基于Python的可交互界面

整合上述两部分代码，生成可交互界面。

```python
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from skimage import morphology
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from natsort import natsorted
import threading
import json

PARAMS_FILE = "params.json"  # 程序参数文件名称。用于储存程序的终止参数，或在程序启动时自动读入以加载为初始参数，生成于工作目录或程序所在文件夹内。

def load_params():
    # 检查参数文件是否存在
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f)
    return None

def save_params():
    # 保存参数文件
    params = {
        "input_dir": input_dir_entry.get(),
        "output_dir": output_dir_entry.get(),
        "pixels_per_cm": pixels_per_cm_entry.get(),
        "crop_top": crop_top_entry.get(),
        "crop_bottom": crop_bottom_entry.get(),
        "crop_left": crop_left_entry.get(),
        "crop_right": crop_right_entry.get(),
        "threshold_method": threshold_method_var.get(),
        "fixed_threshold_value": fixed_threshold_entry.get(),
        "root_background": root_background_var.get(),
        "cpu_threads": cpu_threads_entry.get(),
    }
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params, f)

# 加载初始参数（如果该文件存在的话）
saved_params = load_params()

def create_tooltip(widget, text):
    # 定义程序界面提示条格式
    tooltip = tk.Toplevel(widget, bg="lightyellow", padx=5, pady=5)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = tk.Label(tooltip, text=text, bg="lightyellow")
    label.pack()
    
    def show_tooltip(event):
        tooltip.geometry(f"+{event.x_root + 20}+{event.y_root}")
        tooltip.deiconify()

    def hide_tooltip(event):
        tooltip.withdraw()

    widget.bind("<Enter>", show_tooltip)
    widget.bind("<Leave>", hide_tooltip)

def save_with_increment(filepath, save_function, *args, **kwargs):
    # 分析结果的导出名称。如果存在同名文件，则自动添加编号以区分
    base_name, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base_name}_{counter}{ext}"
        counter += 1
    save_function(new_filepath, *args, **kwargs)

def run_analysis():
    # 图像分析流程
    status_text.config(state="normal")
    status_text.delete(1.0, tk.END)
    
    input_dir = input_dir_entry.get()
    output_dir = output_dir_entry.get()
    pixels_per_cm = int(pixels_per_cm_entry.get())
    crop_top_percentage = float(crop_top_entry.get()) / 100
    crop_bottom_percentage = float(crop_bottom_entry.get()) / 100
    crop_left_percentage = float(crop_left_entry.get()) / 100
    crop_right_percentage = float(crop_right_entry.get()) / 100
    threshold_method = threshold_method_var.get()
    fixed_threshold_value = int(fixed_threshold_entry.get())
    root_background = root_background_var.get()
    cpu_threads = int(cpu_threads_entry.get())
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'root_analysis_results.csv')

    def process_image(image_file):
        image_path = os.path.join(input_dir, image_file)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("无法加载图像")
        except Exception as e:
            status_text.insert(tk.END, f"无法加载图像 {image_file}，跳过\n")
            root.update_idletasks()
            return None

        try:
            height, width = image.shape
            crop_top = int(height * crop_top_percentage)
            crop_bottom = int(height * crop_bottom_percentage)
            crop_left = int(width * crop_left_percentage)
            crop_right = int(width * crop_right_percentage)
            image_cropped = image[crop_top:height - crop_bottom, crop_left:width - crop_right]

            if root_background == "白底黑根":
                if threshold_method == "fixed":
                    _, binary = cv2.threshold(image_cropped, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
                    threshold_value = fixed_threshold_value
                elif threshold_method == "otsu":
                    threshold_value, binary = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif root_background == "黑底白根":
                if threshold_method == "fixed":
                    _, binary = cv2.threshold(image_cropped, fixed_threshold_value, 255, cv2.THRESH_BINARY)
                    threshold_value = fixed_threshold_value
                elif threshold_method == "otsu":
                    threshold_value, binary = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            skeleton = morphology.skeletonize(binary // 255).astype(np.uint8) * 255
            base_name = os.path.splitext(image_file)[0]
            binary_path = os.path.join(output_dir, f'{base_name}_binary.jpg')
            skeleton_path = os.path.join(output_dir, f'{base_name}_skeleton.jpg')
            save_with_increment(binary_path, cv2.imwrite, binary)
            save_with_increment(skeleton_path, cv2.imwrite, skeleton)

            pixel_length = np.sum(skeleton) / 255
            total_root_length_cm = pixel_length / pixels_per_cm
            projected_area = np.sum(binary // 255) / (pixels_per_cm**2)
            average_root_diameter_mm = (projected_area / total_root_length_cm) * 10
            root_surface_area = np.pi * (average_root_diameter_mm / 10) * total_root_length_cm
            root_volume = np.pi * ((average_root_diameter_mm / 2) / 10)**2 * total_root_length_cm

            return {
                "Image": image_file,
                "Threshold Value": threshold_value,
                "Projected Area (cm2)": projected_area,
                "Total Root Length (cm)": total_root_length_cm,
                "Average Root Diameter (mm)": average_root_diameter_mm,
                "Root Surface Area (cm2)": root_surface_area,
                "Root Volume (cm3)": root_volume
            }
        except Exception as e:
            status_text.insert(tk.END, f"处理图像 {image_file} 时出错，跳过\n")
            root.update_idletasks()
            return None

    def background_task():
        results = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=cpu_threads) as executor:
            futures = {executor.submit(process_image, image_file): image_file for image_file in os.listdir(input_dir)}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        end_time = time.time()
        status_text.insert(tk.END, f"多线程并行运算时间: {end_time - start_time:.2f} s\n")
        root.update_idletasks()

        results = natsorted(results, key=lambda x: x["Image"])
        df = pd.DataFrame(results)
        save_with_increment(output_csv, df.to_csv, index=False)
        os.startfile(output_dir)

    threading.Thread(target=background_task).start()

# 创建程序窗口
root = tk.Tk()
root.title("Root Analysis Tool")
root.geometry("370x370")

# 程序初始参数。载入已保存的程序参数，没有则应用内置默认值
input_dir_entry = tk.Entry(root, width=30)
input_dir_entry.insert(0, saved_params.get("input_dir", "D:\\R\\RootAnalysis\\Input") if saved_params else "D:\\R\\RootAnalysis\\Input")
output_dir_entry = tk.Entry(root, width=30)
output_dir_entry.insert(0, saved_params.get("output_dir", "D:\\R\\RootAnalysis\\Output") if saved_params else "D:\\R\\RootAnalysis\\Output")
pixels_per_cm_entry = tk.Entry(root, width=10)
pixels_per_cm_entry.insert(0, saved_params.get("pixels_per_cm", "130") if saved_params else "130")
crop_top_entry = tk.Entry(root, width=10)
crop_top_entry.insert(0, saved_params.get("crop_top", "0") if saved_params else "0")
crop_bottom_entry = tk.Entry(root, width=10)
crop_bottom_entry.insert(0, saved_params.get("crop_bottom", "0") if saved_params else "0")
crop_left_entry = tk.Entry(root, width=10)
crop_left_entry.insert(0, saved_params.get("crop_left", "0") if saved_params else "0")
crop_right_entry = tk.Entry(root, width=10)
crop_right_entry.insert(0, saved_params.get("crop_right", "0") if saved_params else "0")
threshold_method_var = tk.StringVar(value=saved_params.get("threshold_method", "otsu") if saved_params else "otsu")
fixed_threshold_entry = tk.Entry(root, width=10)
fixed_threshold_entry.insert(0, saved_params.get("fixed_threshold_value", "100") if saved_params else "100")
root_background_var = tk.StringVar(value=saved_params.get("root_background", "白底黑根") if saved_params else "白底黑根")
cpu_threads_entry = tk.Entry(root, width=10)
cpu_threads_entry.insert(0, saved_params.get("cpu_threads", "6") if saved_params else "6")

# 程序界面-原始图像位置
input_dir_label = tk.Label(root, text="原始图像位置：")
input_dir_label.grid(row=0, column=0, sticky="e", padx=0)
input_dir_entry.grid(row=0, column=1, sticky="w")
input_dir_button = tk.Button(root, text="选择", command=lambda: input_dir_entry.insert(0, filedialog.askdirectory()))
input_dir_button.grid(row=0, column=2, sticky="w")
create_tooltip(input_dir_label, "原始图像的存放位置")

# 程序界面-结果导出位置
output_dir_label = tk.Label(root, text="结果导出位置：")
output_dir_label.grid(row=1, column=0, sticky="e")
output_dir_entry.grid(row=1, column=1, sticky="w")
output_dir_button = tk.Button(root, text="选择", command=lambda: output_dir_entry.insert(0, filedialog.askdirectory()))
output_dir_button.grid(row=1, column=2, sticky="w")
create_tooltip(output_dir_label, "二值化图像、根骨架图像、结果数据的导出位置")

# 程序界面-校准系数
pixels_per_cm_label = tk.Label(root, text="校准系数：")
pixels_per_cm_label.grid(row=2, column=0, sticky="e")
pixels_per_cm_entry.grid(row=2, column=1, sticky="w")
create_tooltip(pixels_per_cm_label, "像素到厘米的转换比例")

# 程序界面-图像裁剪百分比
crop_top_label = tk.Label(root, text="裁除顶部百分比：")
crop_top_label.grid(row=3, column=0, sticky="e")
crop_top_entry.grid(row=3, column=1, sticky="w")

crop_bottom_label = tk.Label(root, text="裁除底部百分比：")
crop_bottom_label.grid(row=4, column=0, sticky="e")
crop_bottom_entry.grid(row=4, column=1, sticky="w")

crop_left_label = tk.Label(root, text="裁除左部百分比：")
crop_left_label.grid(row=5, column=0, sticky="e")
crop_left_entry.grid(row=5, column=1, sticky="w")

crop_right_label = tk.Label(root, text="裁除右部百分比：")
crop_right_label.grid(row=6, column=0, sticky="e")
crop_right_entry.grid(row=6, column=1, sticky="w")

# 程序界面-图像类型
root_background_label = tk.Label(root, text="图像类型：")
root_background_label.grid(row=7, column=0, sticky="e")
white_radio = ttk.Radiobutton(root, text="白底黑根", variable=root_background_var, value="白底黑根")
white_radio.grid(row=7, column=1, sticky="w")
create_tooltip(white_radio, "阈值越大，根系越厚")
black_radio = ttk.Radiobutton(root, text="黑底白根", variable=root_background_var, value="黑底白根")
black_radio.grid(row=7, column=1, sticky="w", padx=80)
create_tooltip(black_radio, "阈值越小，根系越厚")

# 程序界面-阈值方法
threshold_method_label = tk.Label(root, text="阈值方法：")
threshold_method_label.grid(row=8, column=0, sticky="e")
fixed_radio = ttk.Radiobutton(root, text="fixed", variable=threshold_method_var, value="fixed")
fixed_radio.grid(row=8, column=1, sticky="w")
otsu_radio = ttk.Radiobutton(root, text="otsu", variable=threshold_method_var, value="otsu")
otsu_radio.grid(row=8, column=1, sticky="w", padx=80)
create_tooltip(threshold_method_label, "二值化阈值确定方式")
create_tooltip(fixed_radio, "固定阈值法")
create_tooltip(otsu_radio, "自动阈值法")

# 程序界面-固定阈值
fixed_threshold_label = tk.Label(root, text="固定阈值：")
fixed_threshold_label.grid(row=9, column=0, sticky="e")
fixed_threshold_entry.grid(row=9, column=1, sticky="w")
create_tooltip(fixed_threshold_label, "固定阈值仅在使用固定阈值法时有效")

# 程序界面-CPU线程数
cpu_threads_label = tk.Label(root, text="CPU线程数：")
cpu_threads_label.grid(row=10, column=0, sticky="e")
cpu_threads_entry.grid(row=10, column=1, sticky="w")
create_tooltip(cpu_threads_label, "并行运算调用的CPU线程数上限")

# 程序界面-运行分析按钮
run_button = tk.Button(root, text="运行分析", command=lambda: threading.Thread(target=run_analysis).start())
run_button.grid(row=11, column=0, columnspan=3)

# 程序界面-状态栏（带滚动条）
status_frame = tk.Frame(root)
status_frame.grid(row=12, column=0, columnspan=3, sticky="we")
status_text = tk.Text(status_frame, height=5, wrap="word", state="normal", width=40)
status_text.pack(side="left", fill="both", expand=True)
status_scrollbar = tk.Scrollbar(status_frame, command=status_text.yview)
status_scrollbar.pack(side="right", fill="y")
status_text.config(yscrollcommand=status_scrollbar.set)


# 将保存参数文件函数绑定到窗口关闭事件
root.protocol("WM_DELETE_WINDOW", lambda: [save_params(), root.destroy()])

root.mainloop()

```

# 将Python代码打包为可执行文件

为减小可执行文件（exe文件）的体积，本文采用虚拟环境安装Python代码运行所需的最少库/模块（以避免不必要的模块掺入），并使用UPX压缩。

## 下载UPX

下载[UPX压缩包](https://github.com/upx/upx/releases/latest)并解压于任一文件夹，要求路径不含中文。

## 安装Anaconda

本文利用Anaconda Prompt创建虚拟环境。

自行下载并安装[Anaconda](https://www.anaconda.com/download)软件。安装包也可从镜像网站（如[清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D)）下载。

## 创建并激活虚拟环境

从开始菜单运行“Anaconda Prompt”，输入指令。

### 创建虚拟环境

```         
conda create -n aotu python=3.12
```

在创建过程中回复`y`，成功创建一个名字为aotu，且基于python版本3.12的虚拟环境。

### 激活虚拟环境

```         
conda activate aotu
```

### 查看虚拟环境

```         
conda info --envs
```

```         
conda list
```

### 安装代码运行需要的库

将代码中的库/模块与虚拟环境中已有的进行比对，安装缺失的库。

```         
pip install opencv-python numpy scikit-image pandas natsort
```

同时安装脚本打包模块。

```         
pip install pyinstaller
```

## 创建可执行文件

### 切换路径

切换到待打包的py代码文件所处的文件夹。

```         
D:
cd R\RootAnalysis
```

### 打包

在py代码文件所处的文件夹，新建版本信息文件version_info.txt，并在其中填入以下内容：

```         
# 这里指定文件版本和产品版本为 1.0.0.0
VSVersionInfo(
   ffi=FixedFileInfo(
      filevers=(1, 0, 0, 0),  # 文件版本
      prodvers=(1, 0, 0, 0),  # 产品版本
      mask=0x3f,
      flags=0x0,
      OS=0x4,
      fileType=0x1,
      subtype=0x0,
      date=(0, 0)
   ),
   kids=[
      StringFileInfo(
         [
            StringTable(
               u'040904B0',
               [
                  StringStruct(u'ProductName', u'Root Analysis Tool')
               ])
         ]
      ),
      VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
   ]
)
```

生成可执行文件。继续在Anaconda Prompt激活的虚拟环境中，输入以下指令：

```         
pyinstaller --onefile --noconsole --name root_analysis_tool --version-file D:\R\RootAnalysis\version_info.txt --upx-dir "D:\Program Files\Python312\upx-4.2.4-win64" D:\R\RootAnalysis\root_analysis_tool_package.py
```

其中，`root_analysis_tool`为生成的程序名称，`D:\R\RootAnalysis\version_info.txt`为版本文件所处位置及名称，`D:\Program Files\Python312\upx-4.2.4-win64`为UPX文件所处位置，`D:\R\RootAnalysis\root_analysis_tool_package.py`为py代码所处位置及名称。

生成的可执行文件位于dist文件夹内。

## 退出并清空虚拟环境

### 退出虚拟环境

```         
conda deactivate
```

### 删除虚拟环境

```         
conda remove --name aotu --all
```
