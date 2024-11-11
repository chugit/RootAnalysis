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
