import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
import os
import sys

# 导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.esm_encoder import ESMEncoder
from models.gnn_ppi import GNN_PPI
from models.cot_generator import COTGenerator
from utils.caching import ESMEncodingCache

class SpeedBenchmark:
    """速度基准测试"""
    def __init__(self, output_dir: str = 'benchmark_results'):
        """
        初始化速度基准测试
        
        Args:
            output_dir: 基准测试结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def measure_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """
        测量函数执行时间
        
        Args:
            func: 要测量的函数
            args: 函数参数
            kwargs: 函数关键字参数
        
        Returns:
            执行时间（秒）和函数返回值
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result
    
    def benchmark_esm_encoding(self, sequences: List[str], batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, List[float]]:
        """
        测试ESM编码器的速度
        
        Args:
            sequences: 测试序列列表
            batch_sizes: 批次大小列表
        
        Returns:
            不同批次大小的编码时间
        """
        print("Running ESM encoding benchmark...")
        
        # 初始化ESM编码器
        encoder = ESMEncoder(model_name='esm2_t6_8M_UR50D')
        
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            
            # 将序列分成批次
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch = sequences[start_idx:end_idx]
                
                # 测量编码时间
                encode_time, _ = self.measure_time(encoder.encode, batch)
                times.append(encode_time)
            
            avg_time = np.mean(times)
            results[f'batch_size_{batch_size}'] = avg_time
            print(f"  Batch size {batch_size}: {avg_time:.4f} seconds")
        
        self.results['esm_encoding'] = results
        return results
    
    def benchmark_gnn_forward(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, List[float]]:
        """
        测试GNN前向传播的速度
        
        Args:
            batch_sizes: 批次大小列表
        
        Returns:
            不同批次大小的GNN前向传播时间
        """
        print("Running GNN forward benchmark...")
        
        # 初始化GNN模型
        gnn = GNN_PPI(
            num_features=320,
            hidden_dim=128,
            num_classes=1,
            num_layers=2
        )
        gnn.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            # 创建随机输入数据
            x = torch.randn(batch_size, 100, 320)
            batch = torch.zeros(batch_size * 100, dtype=torch.long)
            
            for i in range(batch_size):
                batch[i*100:(i+1)*100] = i
            
            # 测量前向传播时间
            with torch.no_grad():
                forward_time, _ = self.measure_time(gnn, x, batch)
            
            results[f'batch_size_{batch_size}'] = forward_time
            print(f"  Batch size {batch_size}: {forward_time:.4f} seconds")
        
        self.results['gnn_forward'] = results
        return results
    
    def benchmark_cot_generation(self, sequences: List[str], num_queries: int = 10) -> Dict[str, float]:
        """
        测试COT生成器的速度
        
        Args:
            sequences: 测试序列列表
            num_queries: 查询数量
        
        Returns:
            COT生成时间
        """
        print("Running COT generation benchmark...")
        
        # 初始化COT生成器
        cot_generator = COTGenerator()
        
        times = []
        
        for i in range(num_queries):
            seq1, seq2 = sequences[i % len(sequences)], sequences[(i+1) % len(sequences)]
            
            # 测量COT生成时间
            cot_time, _ = self.measure_time(
                cot_generator.generate_cot,
                seq1, seq2,
                task="ppi_prediction"
            )
            times.append(cot_time)
        
        avg_time = np.mean(times)
        results = {'average_time': avg_time}
        print(f"  Average COT generation time: {avg_time:.4f} seconds")
        
        self.results['cot_generation'] = results
        return results
    
    def benchmark_inference_pipeline(self, sequences: List[str], batch_sizes: List[int] = [1, 4, 8]) -> Dict[str, List[float]]:
        """
        测试完整推理流水线的速度
        
        Args:
            sequences: 测试序列列表
            batch_sizes: 批次大小列表
        
        Returns:
            不同批次大小的推理时间
        """
        print("Running inference pipeline benchmark...")
        
        # 初始化所有组件
        encoder = ESMEncoder(model_name='esm2_t6_8M_UR50D')
        gnn = GNN_PPI(
            num_features=320,
            hidden_dim=128,
            num_classes=1,
            num_layers=2
        )
        gnn.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            
            # 创建蛋白质对批次
            num_pairs = len(sequences) // 2
            num_batches = (num_pairs + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_pairs)
                
                # 获取蛋白质对
                pairs = [(sequences[2*j], sequences[2*j+1]) for j in range(start_idx, end_idx)]
                
                # 测量完整推理时间
                inference_time, _ = self.measure_time(self._run_inference, pairs, encoder, gnn)
                times.append(inference_time)
            
            avg_time = np.mean(times)
            results[f'batch_size_{batch_size}'] = avg_time
            print(f"  Batch size {batch_size}: {avg_time:.4f} seconds")
        
        self.results['inference_pipeline'] = results
        return results
    
    def _run_inference(self, pairs: List[Tuple[str, str]], encoder: ESMEncoder, gnn: GNN_PPI) -> List[float]:
        """
        运行完整推理流程
        
        Args:
            pairs: 蛋白质对列表
            encoder: ESM编码器
            gnn: GNN模型
        
        Returns:
            预测结果列表
        """
        # 提取所有序列
        all_sequences = [seq for pair in pairs for seq in pair]
        
        # 编码序列
        encodings = encoder.encode(all_sequences)
        
        # 将编码分成蛋白质对
        pair_encodings = []
        for i in range(0, len(encodings), 2):
            pair_encodings.append((encodings[i], encodings[i+1]))
        
        # 构建图并运行GNN
        predictions = []
        with torch.no_grad():
            for enc1, enc2 in pair_encodings:
                # 这里简化处理，实际应该构建完整的图结构
                x = torch.cat([enc1, enc2], dim=0).unsqueeze(0)
                batch = torch.zeros(x.size(1), dtype=torch.long)
                
                pred = gnn(x, batch)
                predictions.append(pred.item())
        
        return predictions
    
    def save_results(self, filename: str = 'speed_benchmark_results.json'):
        """
        保存基准测试结果
        
        Args:
            filename: 结果文件名
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def generate_report(self, filename: str = 'speed_benchmark_report.png'):
        """
        生成速度基准测试报告
        
        Args:
            filename: 报告文件名
        """
        print("Generating speed benchmark report...")
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 创建子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Speed Benchmark Report', fontsize=16)
        
        # ESM编码速度
        if 'esm_encoding' in self.results:
            esm_data = self.results['esm_encoding']
            batch_sizes = [int(k.split('_')[-1]) for k in esm_data.keys()]
            times = list(esm_data.values())
            
            axs[0, 0].bar(batch_sizes, times)
            axs[0, 0].set_title('ESM Encoding Speed')
            axs[0, 0].set_xlabel('Batch Size')
            axs[0, 0].set_ylabel('Time (seconds)')
        
        # GNN前向传播速度
        if 'gnn_forward' in self.results:
            gnn_data = self.results['gnn_forward']
            batch_sizes = [int(k.split('_')[-1]) for k in gnn_data.keys()]
            times = list(gnn_data.values())
            
            axs[0, 1].bar(batch_sizes, times)
            axs[0, 1].set_title('GNN Forward Propagation Speed')
            axs[0, 1].set_xlabel('Batch Size')
            axs[0, 1].set_ylabel('Time (seconds)')
        
        # COT生成速度
        if 'cot_generation' in self.results:
            cot_data = self.results['cot_generation']
            
            axs[1, 0].bar(['COT Generation'], [cot_data['average_time']])
            axs[1, 0].set_title('COT Generation Speed')
            axs[1, 0].set_ylabel('Average Time (seconds)')
        
        # 推理流水线速度
        if 'inference_pipeline' in self.results:
            pipeline_data = self.results['inference_pipeline']
            batch_sizes = [int(k.split('_')[-1]) for k in pipeline_data.keys()]
            times = list(pipeline_data.values())
            
            axs[1, 1].bar(batch_sizes, times)
            axs[1, 1].set_title('Inference Pipeline Speed')
            axs[1, 1].set_xlabel('Batch Size')
            axs[1, 1].set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Report saved to {output_path}")


def main():
    """运行所有速度基准测试"""
    # 创建测试序列
    test_sequences = [
        "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG"
        "TTEKTDNLAAQIKNVMGKLKVLGYGHDDKTLKILGHNNGTKGNIPIGAAALIKRKDELYL"
        "KDQVILLNKHIEKDQICRFDDGKYKTFNDMLNLLHAYMRLGLGRYYPKGNLIGHEDRQRI"
        "IGKKGQPMKILEFMRYPEDGHNHLENYHTQPEDQGAEKFNAILhlLPHKNIIPVRKLRAYQ"
        "LRHGGGNLIVKHFKLSDTHKPKGHFDPFVQADHQKEGTVLVKVSMKNGGKLFQETTKAIEA"
        "DNIHRIEKTLKAQGKPEKIWDNIIPGKMNSFIADNSQLDSKLTLMGQFYVMDDKKTVEQVIA"
        "EKEKEFGGKIKIVEFICFEVGEGLEKKTEDFAAEVAAQL"
        for _ in range(32)
    ]
    
    # 创建速度基准测试实例
    benchmark = SpeedBenchmark()
    
    # 运行所有基准测试
    benchmark.benchmark_esm_encoding(test_sequences)
    benchmark.benchmark_gnn_forward()
    benchmark.benchmark_cot_generation(test_sequences[:20])
    benchmark.benchmark_inference_pipeline(test_sequences)
    
    # 保存结果和生成报告
    benchmark.save_results()
    benchmark.generate_report()


if __name__ == "__main__":
    main()

