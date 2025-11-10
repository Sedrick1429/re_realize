#!/bin/bash
# setup.sh - Pose Magic 完整环境配置脚本

set -e  # 遇到错误立即退出

echo "================================"
echo "Pose Magic 环境配置脚本"
echo "================================"

# 检查Python版本
echo "[1/7] 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $PYTHON_VERSION"

#/'''# 创建虚拟环境
#echo "[2/7] 创建虚拟环境..."
#if [ ! -d "venv" ]; then
#    python3 -m venv venv
 #   echo "✓ 虚拟环境创建成功"
#else
 #   echo "✓ 虚拟环境已存在"
#fi

# 激活虚拟环境
#echo "[3/7] 激活虚拟环境..."
#source venv/bin/activate
#echo "✓ 虚拟环境已激活"

# 升级pip
echo "[4/7] 升级pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip已升级"

# [5/7] 安装依赖包...
if [ -f "requirements.txt" ]; then
    echo "正在为 PyTorch 2.3.1 (cu121) 和 Python 3.11 安装GPU依赖..."

    # 使用 python3 -m pip 並儲存退出代碼
    python3 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html \
        --no-cache-dir \
        -r requirements.txt

    # 檢查上一個命令 (pip) 是否成功
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: 依赖包安装失败 (pip install failed)!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # 捕獲到錯誤，立即退出
    fi

    echo "✓ 依赖包安装成功"
fi

# [6/7] 验证安装...
echo "[6/7] 验证安装..."
# [修正] 使用 <<EOF 來避免 shell 語法錯誤和 Python 縮進錯誤
# [FIX] Use <<EOF to avoid shell syntax errors and Python indentation errors
python3 <<EOF
import torch
import numpy as np
import scipy
print('✓ PyTorch版本:', torch.__version__)
print('✓ NumPy版本:', np.__version__)
print('✓ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA设备:', torch.cuda.get_device_name(0))
EOF

# [7/7] 创建项目目录结构...
echo "[7/7] 创建项目目录结构..."
mkdir -p models data checkpoints checkpoints_causal logs utils configs
mkdir -p data/raw data/processed
echo "✓ 目录结构创建成功"

echo ""
echo "================================"
echo "✓ 环境配置完成！"
echo "================================"
echo ""
echo "后续步骤："
echo "1. 激活环境: source venv/bin/activate"
echo "2. 下载数据: 访问 http://vision.imar.ro/human3.6m/"
echo "3. 运行训练: python train.py --data-dir ./data --epochs 90"
echo ""