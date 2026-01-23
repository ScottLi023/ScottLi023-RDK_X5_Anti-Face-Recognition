#!/bin/bash
# 只做一件事：激活虚拟环境（需用 source 调用）
if [ -f "myenv/bin/activate" ]; then
    source myenv/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "❌ 虚拟环境不存在，请先创建"
fi