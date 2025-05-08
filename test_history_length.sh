#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python -u src/main.py ...' \"tag_test_qmix_b\""
    exit 1
fi

base_command=$1
log_file=$2

delay_value=12
# 循环执行n_expand_action从0到12的所有变体
for n_expand_action in $(seq 1 12); do
    echo "DEBUG: Current delay_value = $delay_value"

    # 构造完整命令
    full_command="${base_command} delay_type=\"f\" delay_value=${delay_value} delay_scope=0 n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 打印将要执行的命令
    # echo "Executing: ${full_command}"
    
    # 执行命令
    eval ${full_command}
    
    # 等待上一个命令完成（如果不需要等待可以去掉这一行）
    wait
    
    # 可选：添加一些延迟以防止资源冲突
    sleep 2
done

echo "All commands have been submitted."