#!/bin/bash
# test_mininet.sh

echo "测试不同Python版本的Mininet可用性..."

# 测试Python2
echo "=== Python2 ==="
sudo python2 -c "from mininet.net import Mininet; print('Python2: Mininet可用')" 2>/dev/null || echo "Python2: Mininet不可用"

# 测试系统Python3
echo "=== 系统Python3 ==="
sudo python3 -c "from mininet.net import Mininet; print('系统Python3: Mininet可用')" 2>/dev/null || echo "系统Python3: Mininet不可用"

# 测试conda环境Python
echo "=== Conda Python ==="
sudo /home/user4/anaconda3/envs/leocraft/bin/python3 -c "from mininet.net import Mininet; print('Conda Python: Mininet可用')" 2>/dev/null || echo "Conda Python: Mininet不可用"

echo "测试完成"