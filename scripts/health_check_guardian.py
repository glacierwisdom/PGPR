import os
import re
import time
import signal
import subprocess
from collections import deque

class TrainingGuardian:
    def __init__(self, log_dir="logs", check_interval=30):
        self.log_dir = log_dir
        self.check_interval = check_interval
        self.rewards = deque(maxlen=20)
        self.losses = deque(maxlen=20)
        self.last_epoch = -1
        
        # 阈值设定
        self.REWARD_DROP_THRESHOLD = 0.25  # 奖励回撤 25% 视为异常
        self.OUTPUT_VERBOSITY_LIMIT = 4     # 单次预测输出超过 4 个选项视为逻辑退化 (广撒网)
        self.PATIENCE = 5                  # 容忍异常的连续 Epoch 数
        
        self.anomaly_count = 0
        self.is_running = True

    def get_latest_log(self):
        if not os.path.exists(self.log_dir):
            return None
        # 匹配 root_ 开头的所有文件，包括轮转后的 .log.1, .log.2
        logs = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith("root_")]
        if not logs:
            return None
        # 返回最近修改的文件
        return max(logs, key=os.path.getmtime)

    def parse_log_status(self, log_file):
        status = {
            "epoch": -1,
            "loss": None,
            "reward": None,
            "verbosity": 0
        }
        
        try:
            with open(log_file, 'r', errors='ignore') as f:
                lines = f.readlines()[-200:] # 只看最后 200 行
                
                for line in reversed(lines):
                    # 解析指标 (增加对实时 METRICS 行的支持)
                    metric_match = re.search(r"Epoch (\d+) - 训练完成，损失：([\d\.]+)，奖励：([\d\.]+)", line)
                    step_match = re.search(r"\[METRICS\] Epoch (\d+) Step (\d+): loss=([\d\.]+), reward=([\d\.]+)", line)
                    
                    if metric_match and status["epoch"] == -1:
                        status["epoch"] = int(metric_match.group(1))
                        status["loss"] = float(metric_match.group(2))
                        status["reward"] = float(metric_match.group(3))
                    elif step_match and status["epoch"] == -1:
                        status["epoch"] = int(step_match.group(1))
                        status["loss"] = float(step_match.group(3))
                        status["reward"] = float(step_match.group(4))
                    
                    # 解析输出多样性 (广撒网检测)
                    if "Predicted Output:" in line and status["verbosity"] == 0:
                        output_content = line.split("Predicted Output:")[1]
                        options = [opt.strip() for opt in re.split(r'[,，、]', output_content) if opt.strip()]
                        status["verbosity"] = len(options)
                        
                    if status["epoch"] != -1 and status["verbosity"] != 0:
                        break
        except Exception as e:
            print(f"解析错误: {e}")
            
        return status

    def evaluate_health(self, status):
        if status["epoch"] <= 0:
            return True, "等待数据中..."

        issues = []
        
        # 1. 奖励回撤检查
        if status["reward"] is not None:
            self.rewards.append(status["reward"])
            if len(self.rewards) >= 5:
                avg_reward = sum(list(self.rewards)[:-1]) / (len(self.rewards) - 1)
                if status["reward"] < avg_reward * (1 - self.REWARD_DROP_THRESHOLD):
                    issues.append(f"奖励回撤过大 (当前:{status['reward']:.3f}, 历史均值:{avg_reward:.3f})")

        # 2. 逻辑退化检查 (广撒网)
        if status["verbosity"] > self.OUTPUT_VERBOSITY_LIMIT:
            issues.append(f"检测到逻辑退化: 一次性输出 {status['verbosity']} 个选项，疑似广撒网刷分")

        # 3. 结果判断
        if issues:
            self.anomaly_count += 1
            msg = " | ".join(issues)
            if self.anomaly_count >= self.PATIENCE:
                return False, f"终止训练！连续 {self.anomaly_count} 次检测到异常: {msg}"
            return True, f"警告: 检测到潜在问题 ({self.anomaly_count}/{self.PATIENCE}): {msg}"
        else:
            self.anomaly_count = max(0, self.anomaly_count - 1) # 恢复健康
            return True, f"训练正常 (Epoch {status['epoch']}, Reward: {status['reward']:.3f})"

    def stop_training(self):
        print("\n" + "!"*50)
        print("正在触发熔断机制，停止训练进程...")
        print("!"*50)
        
        # 查找训练进程 (main.py train)
        try:
            cmd = "ps aux | grep 'main.py train' | grep -v grep | awk '{print $2}'"
            pids = subprocess.check_output(cmd, shell=True).decode().split()
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
                print(f"已向进程 {pid} 发送终止信号。")
        except Exception as e:
            print(f"停止进程失败: {e}")
        
        self.is_running = False

    def run(self):
        print("--- 训练看门狗 (Health Guardian) 已启动 ---")
        print(f"监控标准: 奖励回撤 > {self.REWARD_DROP_THRESHOLD*100}%, 输出冗余 > {self.OUTPUT_VERBOSITY_LIMIT} 项")
        
        while self.is_running:
            log_file = self.get_latest_log()
            if log_file:
                status = self.parse_log_status(log_file)
                is_healthy, message = self.evaluate_health(status)
                
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {message}")
                
                if not is_healthy:
                    self.stop_training()
                    break
            else:
                print("等待日志文件生成...")
            
            time.sleep(self.check_interval)

if __name__ == "__main__":
    guardian = TrainingGuardian()
    guardian.run()
