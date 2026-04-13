import torch
import numpy as np
import psutil
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json

# 导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.esm_encoder import ESMEncoder
from models.gnn_ppi import GNN_PPI
from models.cot_generator import COTGenerator
from utils.caching import ESMEncodingCache
from utils.batch_processing import GraphNeighborSampler


class MemoryBenchmark:
    """内存基准测试"""
    def __init__(self, output_dir: str = 'benchmark_results'):
        """
        初始化内存基准测试
        
        Args:
            output_dir: 基准测试结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前进程的内存使用情况
        
        Returns:
            内存使用情况字典
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'rss': mem_info.rss / 1024**2,  # 常驻集大小 (MB)
            'vms': mem_info.vms / 1024**2,  # 虚拟内存大小 (MB)
            'used': process.memory_percent()  # 内存使用率 (%)
        }
    
    def benchmark_esm_memory(self, sequences: List[str], batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Dict[str, float]]:
        """
        测试ESM编码器的内存使用情况
        
        Args:
            sequences: 测试序列列表
            batch_sizes: 批次大小列表
        
        Returns:
            不同批次大小的内存使用情况
        """
        print("Running ESM memory benchmark...")
        
        # 初始化ESM编码器
        encoder = ESMEncoder(model_name='esm2_t6_8M_UR50D')
        
        results = {}
        
        # 测量初始内存使用
        initial_memory = self.get_memory_usage()
        
        for batch_size in batch_sizes:
            # 测量编码前的内存
            pre_encode_memory = self.get_memory_usage()
            
            # 将序列分成批次
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch = sequences[start_idx:end_idx]
                
                # 编码序列
                encodings = encoder.encode(batch)
                
                # 测量编码后的内存
                post_encode_memory = self.get_memory_usage()
                
                # 清理编码结果
                del encodings
                torch.cuda.empty_cache()
            
            # 测量清理后的内存
            post_clean_memory = self.get_memory_usage()
            
            # 计算内存增量
            memory_increase = {
                'rss': post_encode_memory['rss'] - pre_encode_memory['rss'],
                'vms': post_encode_memory['vms'] - pre_encode_memory['vms'],
                'used': post_encode_memory['used'] - pre_encode_memory['used']
            }
            
            results[f'batch_size_{batch_size}'] = memory_increase
            print(f"  Batch size {batch_size}: RSS increase = {memory_increase['rss']:.2f} MB")
        
        self.results['esm_memory'] = results
        return results
    
    def benchmark_gnn_memory(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Dict[str, float]]:
        """
        测试GNN的内存使用情况
        
        Args:
            batch_sizes: 批次大小列表
        
        Returns:
            不同批次大小的内存使用情况
        """
        print("Running GNN memory benchmark...")
        
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
            
            # 测量前向传播前的内存
            pre_forward_memory = self.get_memory_usage()
            
            # 运行前向传播
            with torch.no_grad():
                pred = gnn(x, batch)
            
            # 测量前向传播后的内存
            post_forward_memory = self.get_memory_usage()
            
            # 计算内存增量
            memory_increase = {
                'rss': post_forward_memory['rss'] - pre_forward_memory['rss'],
                'vms': post_forward_memory['vms'] - pre_forward_memory['vms'],
                'used': post_forward_memory['used'] - pre_forward_memory['used']
            }
            
            results[f'batch_size_{batch_size}'] = memory_increase
            print(f"  Batch size {batch_size}: RSS increase = {memory_increase['rss']:.2f} MB")
            
            # 清理
            del x, batch, pred
            torch.cuda.empty_cache()
        
        self.results['gnn_memory'] = results
        return results
    
    def benchmark_caching_memory(self, sequences: List[str], cache_types: List[str] = ['memory', 'disk']) -> Dict[str, Dict[str, float]]:
        """
        测试缓存的内存使用情况
        
        Args:
            sequences: 测试序列列表
            cache_types: 缓存类型列表
        
        Returns:
            不同缓存类型的内存使用情况
        """
        print("Running caching memory benchmark...")
        
        # 初始化ESM编码器
        encoder = ESMEncoder(model_name='esm2_t6_8M_UR50D')
        
        results = {}
        
        for cache_type in cache_types:
            # 创建缓存
            if cache_type == 'memory':
                cache = ESMEncodingCache(cache_type='memory')
            else:
                cache = ESMEncodingCache(cache_type='disk', cache_dir='./cache_test')
            
            # 设置缓存
            encoder.set_cache(cache)
            
            # 测量缓存前的内存
            pre_cache_memory = self.get_memory_usage()
            
            # 编码序列（填充缓存）
            encodings = encoder.encode(sequences[:10])
            
            # 测量缓存后的内存
            post_cache_memory = self.get_memory_usage()
            
            # 计算内存增量
            memory_increase = {
                'rss': post_cache_memory['rss'] - pre_cache_memory['rss'],
                'vms': post_cache_memory['vms'] - pre_cache_memory['vms'],
                'used': post_cache_memory['used'] - pre_cache_memory['used']
            }
            
            results[cache_type] = memory_increase
            print(f"  {cache_type} cache: RSS increase = {memory_increase['rss']:.2f} MB")
            
            # 清理
            del encodings
            torch.cuda.empty_cache()
            
            # 移除缓存
            encoder.remove_cache()
            del cache
        
        self.results['caching_memory'] = results
        return results
    
    def benchmark_batch_processing_memory(self, batch_sizes: List[int] = [1, 4, 8, 16], use_sampling: bool = True) -> Dict[str, Dict[str, float]]:
        """
        测试批处理的内存使用情况
        
        Args:
            batch_sizes: 批次大小列表
            use_sampling: 是否使用邻居采样
        
        Returns:
            不同批次大小的内存使用情况
        """
        print(f"Running batch processing memory benchmark (sampling={use_sampling})...")
        
        # 初始化GNN模型
        gnn = GNN_PPI(
            num_features=320,
            hidden_dim=128,
            num_classes=1,
            num_layers=2
        )
        gnn.eval()
        
        # 初始化邻居采样器
        sampler = GraphNeighborSampler(num_layers=2, num_neighbors=[10, 10])
        
        results = {}
        
        for batch_size in batch_sizes:
            # 创建更大的随机输入数据来模拟大图
            x = torch.randn(batch_size, 200, 320)  # 增加节点数量
            batch = torch.zeros(batch_size * 200, dtype=torch.long)
            
            for i in range(batch_size):
                batch[i*200:(i+1)*200] = i
            
            # 测量处理前的内存
            pre_process_memory = self.get_memory_usage()
            
            if use_sampling:
                # 使用邻居采样（这里简化处理）
                with torch.no_grad():
                    pred = gnn(x[:, :100, :], batch[:batch_size * 100])  # 只使用部分节点
            else:
                # 不使用采样
                with torch.no_grad():
                    pred = gnn(x, batch)
            
            # 测量处理后的内存
            post_process_memory = self.get_memory_usage()
            
            # 计算内存增量
            memory_increase = {
                'rss': post_process_memory['rss'] - pre_process_memory['rss'],
                'vms': post_process_memory['vms'] - pre_process_memory['vms'],
                'used': post_process_memory['used'] - pre_process_memory['used']
            }
            
            results[f'batch_size_{batch_size}'] = memory_increase
            print(f"  Batch size {batch_size}: RSS increase = {memory_increase['rss']:.2f} MB")
            
            # 清理
            del x, batch, pred
            torch.cuda.empty_cache()
        
        self.results['batch_processing_memory'] = results
        return results
    
    def save_results(self, filename: str = 'memory_benchmark_results.json'):
        """
        保存基准测试结果
        
        Args:
            filename: 结果文件名
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def generate_report(self, filename: str = 'memory_benchmark_report.png'):
        """
        生成内存基准测试报告
        
        Args:
            filename: 报告文件名
        """
        print("Generating memory benchmark report...")
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 创建子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Memory Benchmark Report', fontsize=16)
        
        # ESM内存使用
        if 'esm_memory' in self.results:
            esm_data = self.results['esm_memory']
            batch_sizes = [int(k.split('_')[-1]) for k in esm_data.keys()]
            rss_values = [v['rss'] for v in esm_data.values()]
            
            axs[0, 0].bar(batch_sizes, rss_values)
            axs[0, 0].set_title('ESM Memory Usage')
            axs[0, 0].set_xlabel('Batch Size')
            axs[0, 0].set_ylabel('RSS Increase (MB)')
        
        # GNN内存使用
        if 'gnn_memory' in self.results:
            gnn_data = self.results['gnn_memory']
            batch_sizes = [int(k.split('_')[-1]) for k in gnn_data.keys()]
            rss_values = [v['rss'] for v in gnn_data.values()]
            
            axs[0, 1].bar(batch_sizes, rss_values)
            axs[0, 1].set_title('GNN Memory Usage')
            axs[0, 1].set_xlabel('Batch Size')
            axs[0, 1].set_ylabel('RSS Increase (MB)')
        
        # 缓存内存使用
        if 'caching_memory' in self.results:
            caching_data = self.results['caching_memory']
            cache_types = list(caching_data.keys())
            rss_values = [v['rss'] for v in caching_data.values()]
            
            axs[1, 0].bar(cache_types, rss_values)
            axs[1, 0].set_title('Caching Memory Usage')
            axs[1, 0].set_ylabel('RSS Increase (MB)')
        
        # 批处理内存使用
        if 'batch_processing_memory' in self.results:
            batch_data = self.results['batch_processing_memory']
            batch_sizes = [int(k.split('_')[-1]) for k in batch_data.keys()]
            rss_values = [v['rss'] for v in batch_data.values()]
            
            axs[1, 1].bar(batch_sizes, rss_values)
            axs[1, 1].set_title('Batch Processing Memory Usage')
            axs[1, 1].set_xlabel('Batch Size')
            axs[1, 1].set_ylabel('RSS Increase (MB)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Report saved to {output_path}")


def main():
    """运行所有内存基准测试"""
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
    
    # 创建内存基准测试实例
    benchmark = MemoryBenchmark()
    
    # 运行所有基准测试
    benchmark.benchmark_esm_memory(test_sequences)
    benchmark.benchmark_gnn_memory()
    benchmark.benchmark_caching_memory(test_sequences)
    benchmark.benchmark_batch_processing_memory()
    
    # 保存结果和生成报告
    benchmark.save_results()
    benchmark.generate_report()


if __name__ == "__main__":
    main()
