#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# 检查root权限
if os.geteuid() != 0:
    print("❌ Mininet必须使用root权限运行")
    sys.exit(1)


# 设置日志记录
def setup_logging():
    """设置详细的日志记录"""
    # 创建日志目录
    log_dir = "/tmp/mininet_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mininet_test_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('MininetTest')
    logger.info(f"日志文件位置: {log_file}")

    return logger, log_file


# 初始化日志
logger, log_file = setup_logging()

# 清理环境
logger.info("清理Mininet环境")
subprocess.run(['pkill', '-f', 'ovs-testcontroller'], check=False)
subprocess.run(['pkill', '-f', 'controller'], check=False)
subprocess.run(['pkill', '-f', 'iperf'], check=False)
subprocess.run(['mn', '-c'], check=False)
time.sleep(2)

try:
    from mininet.net import Mininet
    from mininet.node import OVSController
    from mininet.cli import CLI
    from mininet.log import setLogLevel, info, debug, error, warn
    from mininet.link import TCLink
    from mininet.util import dumpNodeConnections

    logger.info("Mininet模块导入成功")
except ImportError as e:
    logger.error(f"导入Mininet模块失败: {e}")
    sys.exit(1)

# 设置Mininet的日志级别为最详细
setLogLevel('debug')


class LoggingCLI:
    """带日志记录的CLI包装器"""

    def __init__(self, net, logger):
        self.net = net
        self.logger = logger

    def start(self):
        self.logger.info("启动Mininet CLI")
        try:
            CLI(self.net)
        except Exception as e:
            self.logger.error(f"CLI错误: {e}")
        finally:
            self.logger.info("退出Mininet CLI")


def log_network_state(net, logger):
    """记录网络状态"""
    logger.info("=== 网络状态快照 ===")

    # 记录节点连接
    logger.info("节点连接信息:")
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)

    # 记录主机信息
    logger.info("主机信息:")
    for host in net.hosts:
        logger.info(f"  {host.name}: IP={host.IP()}, MAC={host.MAC()}")
        # 记录接口信息
        for intf in host.intfList():
            logger.info(f"    {intf.name}: {intf}")

    # 记录交换机信息
    logger.info("交换机信息:")
    for switch in net.switches:
        logger.info(f"  {switch.name}")
        # 获取OVS信息
        try:
            result = subprocess.run(['ovs-vsctl', 'show'], capture_output=True, text=True)
            logger.debug(f"OVS状态:\n{result.stdout}")
        except Exception as e:
            logger.warning(f"获取OVS信息失败: {e}")


def run_iperf_with_logging(host1, host2, test_type='TCP', duration=5, bw=None, logger=None):
    """运行iperf测试并详细记录"""
    logger.info(f"开始 {test_type} 吞吐量测试 {host1.name} -> {host2.name}")

    # 在host2上启动iperf服务器
    if test_type == 'TCP':
        server_cmd = 'iperf -s'
    else:
        server_cmd = 'iperf -s -u'

    logger.info(f"在 {host2.name} 上启动iperf服务器: {server_cmd}")
    server_proc = host2.popen(server_cmd)
    time.sleep(2)

    # 在host1上运行iperf客户端
    if test_type == 'TCP':
        client_cmd = f'iperf -c {host2.IP()} -t {duration} -i 1'
    else:
        bw_param = f'-b {bw}' if bw else '-b 1M'
        client_cmd = f'iperf -c {host2.IP()} -u {bw_param} -t {duration} -i 1'

    logger.info(f"在 {host1.name} 上运行iperf客户端: {client_cmd}")

    try:
        start_time = time.time()
        client_output = host1.cmd(client_cmd, timeout=duration + 10)
        end_time = time.time()

        logger.info(f"{test_type}测试完成，耗时: {end_time - start_time:.2f}秒")
        logger.info("iperf原始输出:\n" + client_output)

        # 解析并记录关键指标
        lines = client_output.split('\n')
        for line in lines:
            if 'bits/sec' in line and 'sec' in line:
                logger.info(f"📊 带宽结果: {line.strip()}")
            if 'lost' in line.lower():
                logger.info(f"📊 丢包信息: {line.strip()}")

    except Exception as e:
        logger.error(f"{test_type}测试失败: {e}")
    finally:
        server_proc.terminate()
        server_proc.wait()
        time.sleep(1)


def detailed_topology():
    """详细的拓扑测试，包含完整日志记录"""
    logger.info("创建网络拓扑")

    net = Mininet(controller=OVSController, link=TCLink)

    logger.info("添加控制器")
    net.addController('c0')

    logger.info("添加主机")
    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')

    logger.info("添加交换机")
    s1 = net.addSwitch('s1')

    logger.info("创建链路")
    net.addLink(h1, s1, bw=10)
    net.addLink(h2, s1, bw=10)

    logger.info("启动网络")
    net.start()

    # 记录初始网络状态
    log_network_state(net, logger)

    logger.info("基本连通性测试")
    ping_result = net.pingAll()
    logger.info(f"Ping测试结果: {ping_result}")

    # 详细的iperf测试
    logger.info("开始TCP吞吐量测试")
    run_iperf_with_logging(h1, h2, 'TCP', duration=5, logger=logger)

    logger.info("开始UDP吞吐量测试")
    run_iperf_with_logging(h1, h2, 'UDP', duration=5, bw='5M', logger=logger)

    # 测试后再次记录网络状态
    log_network_state(net, logger)

    logger.info("进入交互式CLI")
    print(f"\n💡 详细日志正在记录到: {log_file}")
    print("在CLI中执行的命令也会被记录")

    # 使用带日志的CLI
    LoggingCLI(net, logger).start()

    logger.info("停止网络")
    net.stop()

    logger.info("测试完成")


def capture_packets(net, logger, duration=10):
    """使用tcpdump捕获数据包"""
    logger.info(f"开始数据包捕获，持续时间: {duration}秒")

    # 在所有主机上启动tcpdump
    tcpdump_procs = []
    for host in net.hosts:
        pcap_file = f"/tmp/{host.name}.pcap"
        cmd = f"tcpdump -i any -w {pcap_file} &"
        logger.info(f"在 {host.name} 上启动tcpdump: {cmd}")
        proc = host.popen(cmd)
        tcpdump_procs.append((host, proc, pcap_file))

    logger.info(f"等待 {duration} 秒进行数据包捕获...")
    time.sleep(duration)

    # 停止tcpdump进程
    for host, proc, pcap_file in tcpdump_procs:
        proc.terminate()
        proc.wait()
        logger.info(f"{host.name} 的数据包已保存到: {pcap_file}")

    return tcpdump_procs


def comprehensive_test():
    """综合测试，包含数据包捕获"""
    logger.info("开始综合测试")

    net = Mininet(controller=OVSController, link=TCLink)
    net.addController('c0')

    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')
    s1 = net.addSwitch('s1')

    net.addLink(h1, s1, bw=10)
    net.addLink(h2, s1, bw=10)

    net.start()
    log_network_state(net, logger)

    # 数据包捕获测试
    logger.info("开始带数据包捕获的测试")
    tcpdump_procs = capture_packets(net, logger, duration=5)

    # 在捕获期间运行流量
    logger.info("在数据包捕获期间生成测试流量")
    h2.cmd('iperf -s &')
    time.sleep(1)
    h1.cmd('iperf -c 10.0.0.2 -t 3 &')
    time.sleep(5)  # 等待流量完成

    # 停止捕获
    for host, proc, pcap_file in tcpdump_procs:
        proc.terminate()
        logger.info(f"{host.name} 的pcap文件: {pcap_file}")

    h2.cmd('pkill iperf')

    logger.info("进入CLI进行手动测试")
    LoggingCLI(net, logger).start()

    net.stop()
    logger.info("综合测试完成")


if __name__ == '__main__':
    try:
        print("选择测试模式:")
        print("1. 详细日志测试 (推荐)")
        print("2. 综合测试 (包含数据包捕获)")
        print("3. 最小测试 (仅基本日志)")

        choice = input("请输入选择 (1, 2 或 3): ").strip()

        if choice == '1':
            detailed_topology()
        elif choice == '2':
            comprehensive_test()
        else:
            # 最小测试
            setLogLevel('info')
            net = Mininet(controller=OVSController, link=TCLink)
            net.addController('c0')
            h1 = net.addHost('h1', ip='10.0.0.1/24')
            h2 = net.addHost('h2', ip='10.0.0.2/24')
            s1 = net.addSwitch('s1')
            net.addLink(h1, s1)
            net.addLink(h2, s1)
            net.start()
            logger.info("网络启动完成")
            net.pingAll()
            CLI(net)
            net.stop()

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        logger.info("脚本执行结束")
        print(f"\n📄 详细日志已保存到: {log_file}")