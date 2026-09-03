#!/bin/bash
# 后台训练启动脚本（退出 SSH / 关窗口也不会中断）
#
# 前提：先自己 cd 到仓库根目录 /home/sjx/work/project/DDaTR 并激活 ddatr 环境
# 用法（和你以前一样，只是脚本名换成 train_bg.sh）：
#   nohup bash train_bg.sh > train.txt 2>&1 &
#
# 说明：
#   - 内部用 setsid 把训练放进独立会话，与当前终端彻底脱离，
#     所以关终端 / 断 SSH 都不会影响训练
#   - 查看进度：tail -f train.txt
#   - 停止训练：pkill -f main_train.py
#   - 已有训练在跑时会拒绝重复启动

if pgrep -f "[m]ain_train.py" >/dev/null 2>&1; then
    echo "training already running, check train.txt / train.pid"
    exit 1
fi

# setsid --fork：fork 后进入新会话并脱离终端；PID 写入 train.pid
setsid --fork bash -c 'echo $$ > train.pid; exec bash train_mimic_cxr.sh' >> train.txt 2>&1 < /dev/null &
sleep 1
echo "training started: PID $(cat train.pid 2>/dev/null || echo '?'), log: train.txt"
