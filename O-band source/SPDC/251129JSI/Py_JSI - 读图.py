import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, re, os

# ================= 配置 =================
FILES = ["SFG20251029105128.txt", "SFG20251029180550.txt"]

def robust_load_and_plot(files):
    all_data = []
    
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e'] # 蓝色和橙色区分不同文件
    
    print(">>> 开始诊断性读取数据...")
    
    for i, fpath in enumerate(files):
        if not os.path.exists(fpath):
            print(f"❌ 错误: 找不到文件 {fpath}")
            continue
            
        print(f"--- 读取文件: {fpath} ---")
        
        # 1. 鲁棒读取：跳过坏行
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            # 只读取以数字或符号开头的行
            lines = [ln for ln in f.readlines() if re.match(r'^\s*[+-]?\d', ln)]
            
        if not lines:
            print("   ⚠️ 警告: 文件为空或无数据行")
            continue
            
        # 使用 genfromtxt 自动处理缺失列，用 NaN 填充
        try:
            arr = np.genfromtxt(io.BytesIO(''.join(lines).encode()), delimiter=None)
        except Exception as e:
            print(f"   ❌ 解析错误: {e}")
            continue

        if arr.ndim == 1: arr = arr.reshape(1, -1)
        df = pd.DataFrame(arr)
        
        # 2. 补齐列 (确保至少有9列)
        required_cols = 9
        if df.shape[1] < required_cols:
            print(f"   ℹ️ 提示: 列数不足 ({df.shape[1]}), 自动补全 NaN")
            for c in range(df.shape[1], required_cols):
                df[c] = np.nan
        
        # 3. 数据处理 (Drop 50 + Four-step)
        file_points = []
        
        # 按波长分组 (第0列)
        grouped = df.groupby(0)
        
        for wl, group in grouped:
            # Drop 50
            valid = group.iloc[50:]
            if len(valid) < 5: continue
            
            # 提取数据 (使用 iloc 安全提取)
            # 0:wl, 1:p2, 3:p4, 4:sfg, 6:bg00, 7:bg01, 8:bg10
            vals = valid.values
            
            # 安全获取各列，NaN 转 0
            def get_col(idx):
                col = vals[:, idx]
                return np.nan_to_num(col, nan=0.0) # 关键修正：NaN -> 0
            
            sfg  = get_col(4)
            bg00 = get_col(6)
            bg01 = get_col(7)
            bg10 = get_col(8)
            p_idl = get_col(1)
            p_sig = get_col(3)
            
            # 计算分子 (四步相减)
            num = np.mean(sfg - bg01 - bg10 + bg00)
            
            # 计算分母 (功率乘积)
            mean_p_idl = np.mean(p_idl)
            mean_p_sig = np.mean(p_sig)
            
            # 过滤低功率点 (防止除以零爆炸)
            if abs(mean_p_idl * mean_p_sig) < 1e-6:
                continue
                
            den = mean_p_idl * mean_p_sig
            intensity = num / den
            
            file_points.append([wl, intensity])
            
        # 4. 绘制当前文件的数据
        if file_points:
            file_data = np.array(file_points)
            # 排序
            file_data = file_data[file_data[:,0].argsort()]
            
            # 去除极端异常值 (去底噪) 用于绘图
            # y_clean = file_data[:, 1]
            # y_clean[y_clean < 0] = 0
            
            plt.plot(file_data[:, 0], file_data[:, 1], 'o', markersize=2, 
                     color=colors[i % len(colors)], alpha=0.6, 
                     label=f'File {i+1}: {os.path.basename(fpath)}')
            
            all_data.extend(file_points)
            print(f"   ✅ 成功提取 {len(file_points)} 个波长点")
        else:
            print("   ⚠️ 警告: 该文件未提取到有效数据 (可能功率过低或点数不足)")

    # 5. 设置绘图样式
    if all_data:
        all_arr = np.array(all_data)
        # 归一化用于展示峰值位置
        peak_y = np.max(all_arr[:, 1])
        
        plt.xlabel('Idler Wavelength (nm)', fontsize=12)
        plt.ylabel('Normalized Intensity (a.u.)', fontsize=12)
        plt.title('Experimental Data Diagnostic Plot (Linear Scale)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 自动调整显示范围 (去除底噪的极低负值影响显示)
        # 找到最大值，y轴下限设为最大值的 -5%
        plt.ylim(peak_y * -0.05, peak_y * 1.1)
        
        plt.tight_layout()
        plt.show()
        
        print("-" * 40)
        print(f"峰值强度: {peak_y:.4e}")
        peak_idx = np.argmax(all_arr[:, 1])
        print(f"峰值波长: {all_arr[peak_idx, 0]:.3f} nm")
    else:
        print("\n❌ 严重错误: 没有绘图数据。请检查文件内容格式。")

robust_load_and_plot(FILES)