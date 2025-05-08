#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python -u src/main.py ...' \"tag_test_qmix_b\""
    exit 1
fi

base_command=$1
log_file=$2

# 分为集内和集外
for i in $(seq 0 1); do
    echo "DEBUG: Current i = $i"
    if [ $i -eq 0 ]; then
        delay_value=6
        delay_scope=3
    else
        delay_value=9
        delay_scope=3
    fi
    # 根据log_file和delay_value设置n_expand_action的值
    if [ "$log_file" != "${log_file%ss_h*}" ]; then
        # 如果log_file包含ss_h，则设置最大的n_expand_action，（MPE中最大为9，SMAC中最大为6）
        n_expand_action=9
    elif [ "$log_file" != "${log_file%ms_h*}" ]; then
        # 如果log_file包含ms_h，则设置训练时相同的n_expand_action，（MPE中最大为9，SMAC中最大为6）
        n_expand_action=9
    else
        # 如果log_file既不包含ss_h也不包含ms_h，则n_expand_action=delay_value
        n_expand_action=$((delay_value+delay_scope))
    fi

    # 构造完整命令
    full_command="${base_command} delay_type=\"uf\" delay_value=${delay_value} delay_scope=${delay_scope} n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
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