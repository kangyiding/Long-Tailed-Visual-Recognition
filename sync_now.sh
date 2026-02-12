#!/bin/bash

# 获取当前时间
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "========================================"
echo "🔄 开始手动同步 - $TIMESTAMP"
echo "========================================"

# 1. 添加所有更改
echo "📦 添加文件变更..."
git add .

# 2. 提交更改
# 检查是否有需要提交的更改
if git diff-index --quiet HEAD --; then
    echo "ℹ️  本地没有新的代码变更需要提交。"
else
    echo "💾 提交更改..."
    git commit -m "Manual sync: $TIMESTAMP"
fi

# 3. 推送到远程
echo "🚀 推送到 GitHub..."
git push origin master

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "✅ 同步成功！"
    echo "========================================"
else
    echo "========================================"
    echo "❌ 同步失败，请检查网络或凭证。"
    echo "========================================"
fi
