import os
import subprocess
import argparse
import pandas as pd
import logging
from pathlib import Path
import yaml

import sys
import os

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Benchmark")

class PPIBenchmarkRunner:
    def __init__(self, config_path="configs/base.yaml"):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.absolute()
        self.data_dir = self.project_root / "data" / "processed"
        self.results = {}
        self.python_executable = sys.executable

    def run_command(self, cmd, cwd=None):
        logger.info(f"执行命令: {' '.join(cmd)}")
        # 使用 Popen 来实现实时输出
        process = subprocess.Popen(
            cmd, 
            cwd=cwd or self.project_root, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        full_output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line_str = line.strip()
                print(f"  [SUB] {line_str}", flush=True)
                logger.info(f"[SUB] {line_str}")
                full_output.append(line)
        
        return_code = process.poll()
        if return_code != 0:
            logger.error(f"命令失败 (退出码 {return_code})")
            return False, "".join(full_output)
        return True, "".join(full_output)

    def split_data(self, strategy):
        logger.info(f"正在进行数据划分: {strategy}")
        cmd = [
            self.python_executable, "data/splitter.py",
            "--config", self.config_path,
            "--strategy", strategy,
            "--dataset", "shs27k"
        ]
        success, output = self.run_command(cmd)
        if success:
            logger.info(f"{strategy} 划分成功")
        return success

    def train_and_eval(self, strategy):
        logger.info(f"正在为策略 {strategy} 启动训练和评估...")
        
        # 确定对应的划分文件后缀
        suffix = f"_{strategy}" if strategy != "stratified" else ""
        if strategy == "by_protein":
            suffix = "_by_protein"
            
        train_file = f"shs27k_train{suffix}.tsv"
        val_file = f"shs27k_val{suffix}.tsv"
        test_file = f"shs27k_test{suffix}.tsv"

        # 运行训练
        # 注意：我们通过 config-override 覆盖数据文件
        train_cmd = [
            self.python_executable, "experiments/run_training.py",
            "--config", self.config_path,
            "--config-override",
            f"dataset.train_file={train_file}",
            f"dataset.val_file={val_file}",
            f"dataset.test_file={test_file}",
            "training.epochs=2",  # 跑2轮用于验证流程
            "training.validate=true"
        ]
        
        success, output = self.run_command(train_cmd)
        if not success:
            logger.error(f"{strategy} 训练失败")
            return False

        # 提取评估结果 (这里假设日志中打印了最终指标，或者我们运行 run_evaluation.py)
        # 为了更准确，我们运行专门的评估脚本
        # 首先找到最新的 checkpoint
        checkpoint_dir = self.project_root / "checkpoints"
        # 查找所有 .pth 文件，并根据修改时间排序
        checkpoints = sorted(list(checkpoint_dir.glob("*.pth")), key=os.path.getmtime)
        if not checkpoints:
            # 尝试查找 .pt 文件作为备选
            checkpoints = sorted(list(checkpoint_dir.glob("*.pt")), key=os.path.getmtime)
        
        # 如果还是没找到，尝试在 shs27k 子目录下查找
        if not checkpoints:
            checkpoint_dir_shs = checkpoint_dir / "shs27k"
            if checkpoint_dir_shs.exists():
                checkpoints = sorted(list(checkpoint_dir_shs.glob("*.pth")), key=os.path.getmtime)
                if not checkpoints:
                    checkpoints = sorted(list(checkpoint_dir_shs.glob("*.pt")), key=os.path.getmtime)
        
        if not checkpoints:
            logger.error(f"未找到 {strategy} 的检查点")
            return False
        
        latest_checkpoint = checkpoints[-1]
        logger.info(f"使用最新的检查点: {latest_checkpoint}")

        # 运行评估
        eval_cmd = [
            self.python_executable, "experiments/run_evaluation.py",
            "--config", self.config_path,
            "--checkpoint", str(latest_checkpoint),
            "--data-file", str(self.data_dir / test_file)
        ]
        
        success, output = self.run_command(eval_cmd)
        if success:
            logger.info(f"{strategy} 评估完成")
            # 这里可以从 output 中解析指标
            self.results[strategy] = self._parse_metrics(output)
            return True
        return False

    def _parse_metrics(self, output):
        # 简单解析输出中的指标 (假设 PPIEvaluator 打印了它们)
        metrics = {}
        lines = output.split('\n')
        for line in lines:
            if "accuracy:" in line.lower():
                metrics['accuracy'] = line.split(':')[-1].strip()
            elif "micro_f1:" in line.lower():
                metrics['micro_f1'] = line.split(':')[-1].strip()
            elif "macro_f1:" in line.lower():
                metrics['macro_f1'] = line.split(':')[-1].strip()
            elif "auc:" in line.lower():
                metrics['auc'] = line.split(':')[-1].strip()
            elif "auprc:" in line.lower():
                metrics['auprc'] = line.split(':')[-1].strip()
        return metrics

    def generate_report(self):
        logger.info("生成最终对比报告...")
        df = pd.DataFrame(self.results).T
        report_path = self.project_root / "benchmark_results.csv"
        df.to_csv(report_path)
        
        print("\n" + "="*50)
        print("实验对比结果 (Benchmark Results)")
        print("="*50)
        print(df)
        print("="*50)
        print(f"详细结果已保存至: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="PPI 性能基准测试运行脚本")
    parser.add_argument("--strategies", nargs="+", default=["bs", "es", "ns"], help="要测试的划分策略")
    parser.add_argument("--config", default="configs/base.yaml", help="基础配置路径")
    args = parser.parse_args()

    runner = PPIBenchmarkRunner(args.config)
    
    for strategy in args.strategies:
        logger.info(f"\n>>> 处理策略: {strategy}")
        if runner.split_data(strategy):
            runner.train_and_eval(strategy)
    
    runner.generate_report()

if __name__ == "__main__":
    main()
