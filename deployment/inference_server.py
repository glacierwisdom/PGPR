import os
import sys
import time
import argparse
import logging
import threading
import queue
import hashlib
import torch
from typing import List, Dict, Any, Tuple, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from graph.builder import PPIGraphBuilder
from evaluation.evaluator import PPIEvaluator

os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'inference_server.log'))
    ]
)
logger = logging.getLogger('inference_server')

class InferenceRequest:
    """
    推理请求类，用于封装单个推理请求
    """
    def __init__(self, request_id: str, protein_a: str, protein_b: str, request_time: float):
        self.request_id = request_id
        self.protein_a = protein_a
        self.protein_b = protein_b
        self.request_time = request_time
        self.response = None
        self.completion_time = None

class PPIInferenceServer:
    """
    PGPR 模型的推理服务器
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, num_workers: int = 4, batch_size: int = 32):
        """
        初始化推理服务器
        
        Args:
            config_path (str): 配置文件路径
            checkpoint_path (str): 模型检查点路径
            num_workers (int): 工作线程数量
            batch_size (int): 批处理大小
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # 加载配置
        self.config_manager = ConfigManager()
        base_dir = os.path.join(project_root, 'configs')
        for name in ['base.yaml', 'model.yaml', 'training.yaml', 'data.yaml']:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                self.config_manager.merge_config(self.config_manager.load_config(p))

        self.config_manager.merge_config(self.config_manager.load_config(config_path))
        
        self.config = self.config_manager.config
        dataset_cfg = self.config.get('dataset', {}) or {}
        dataset_name = str(dataset_cfg.get('name', '')).lower()
        if dataset_name in ("shs27k", "shs148k"):
            processed_dir = f"data/{dataset_name}_llapa/processed"
            dataset_cfg.setdefault('train_file', f"{processed_dir}/random_train.tsv")
            dataset_cfg.setdefault('val_file', f"{processed_dir}/random_val.tsv")
            dataset_cfg.setdefault('test_file', f"{processed_dir}/random_test.tsv")
            dataset_cfg.setdefault('protein_info_file', 'protein_info.csv')
            self.config['dataset'] = dataset_cfg
        
        # 设置设备
        self.device = self.config.get('device', {}).get('device_type', 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU")
            self.device = 'cpu'
            # 更新配置中的设备设置
            if 'device' not in self.config:
                self.config['device'] = {}
            self.config['device']['device_type'] = self.device
        
        # 加载模型和相关组件
        self._load_model()
        
        # 初始化请求队列
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 工作线程池
        self.workers = []
        self.is_running = False
        
        # 统计信息
        self.request_counter = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        
        logger.info(f"推理服务器初始化完成，使用设备: {self.device}")
    
    def _load_model(self):
        """
        加载模型和相关组件
        """
        logger.info("开始加载模型...")
        start_time = time.time()

        self.evaluator = PPIEvaluator(self.config)
        self.evaluator.load_model(self.checkpoint_path)

        from models.component_builder import ComponentBuilder
        self.graph_builder = ComponentBuilder.build_graph_builder(self.config)

        train_file = (self.config.get('dataset', {}) or {}).get('train_file')
        if not train_file:
            raise ValueError("配置中缺少 dataset.train_file")
        train_file_abs = os.path.join(project_root, train_file) if not os.path.isabs(train_file) else train_file
        self.graph_data, _ = self.graph_builder.build_graph_and_load_data(
            split="train",
            data_file=train_file_abs,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.protein_info = {}
        try:
            pd = getattr(self.evaluator, "prompt_designer", None)
            if pd is not None and isinstance(getattr(pd, "protein_id_to_info", None), dict):
                self.protein_info = pd.protein_id_to_info
        except Exception:
            self.protein_info = {}
        for pid in getattr(self.graph_data, "protein_ids", []) or []:
            if pid not in self.protein_info:
                self.protein_info[pid] = {'name': str(pid), 'function': '暂无功能描述'}

        loading_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {loading_time:.2f}秒")
    
    def start(self):
        """
        启动推理服务器
        """
        if self.is_running:
            logger.warning("推理服务器已经在运行中")
            return
        
        self.is_running = True
        
        # 启动工作线程
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f'Worker-{i}')
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            logger.info(f"已启动工作线程: Worker-{i}")
        
        # 启动结果处理线程
        self.result_processor = threading.Thread(target=self._result_processor_loop, name='ResultProcessor')
        self.result_processor.daemon = True
        self.result_processor.start()
        
        logger.info("推理服务器已启动")
    
    def stop(self):
        """
        停止推理服务器
        """
        if not self.is_running:
            logger.warning("推理服务器已经停止")
            return
        
        logger.info("正在停止推理服务器...")
        self.is_running = False
        
        # 等待所有工作线程完成
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # 清空队列
        while not self.request_queue.empty():
            self.request_queue.get()
        
        while not self.result_queue.empty():
            self.result_queue.get()
        
        self.workers = []
        logger.info("推理服务器已停止")
    
    def _worker_loop(self):
        """
        工作线程循环，处理推理请求
        """
        while self.is_running:
            try:
                # 从队列获取请求，超时时间为1秒
                request = self.request_queue.get(timeout=1.0)
                
                # 处理请求
                try:
                    self._process_request(request)
                    self.result_queue.put(request)
                except Exception as e:
                    logger.error(f"处理请求 {request.request_id} 时出错: {str(e)}", exc_info=True)
                    request.response = {
                        'status': 'error',
                        'message': str(e),
                        'request_id': request.request_id
                    }
                    request.completion_time = time.time()
                    self.result_queue.put(request)
                finally:
                    self.request_queue.task_done()
            
            except queue.Empty:
                continue
            
            except Exception as e:
                logger.error(f"工作线程出错: {str(e)}", exc_info=True)
    
    def _process_request(self, request: InferenceRequest):
        """
        处理单个推理请求
        
        Args:
            request (InferenceRequest): 推理请求
        """
        start_time = time.time()
        logger.info(f"开始处理请求 {request.request_id}")
        seq_a = request.protein_a
        seq_b = request.protein_b

        pid_a = self._sequence_to_id(seq_a)
        pid_b = self._sequence_to_id(seq_b)

        pid_to_idx = getattr(self.graph_data, "protein_id_to_idx", {}) or {}
        if pid_a not in pid_to_idx or pid_b not in pid_to_idx:
            raise ValueError("输入序列不在当前训练图节点集中；如需支持未见蛋白，请开启/实现 similarity_mapping 的代理映射推理流程")

        with torch.no_grad():
            pred = self.evaluator.evaluate_single_pair(
                model=self.evaluator.cot_generator,
                graph_data=self.graph_data,
                llm_wrapper=self.evaluator.llm_wrapper,
                protein_a=seq_a,
                protein_b=seq_b,
                protein_a_id=pid_a,
                protein_b_id=pid_b,
            )

        response = {
            'status': 'success',
            'request_id': request.request_id,
            'result': pred,
            'processing_time': time.time() - start_time
        }
        request.response = response
        request.completion_time = time.time()
        self._update_stats(time.time() - start_time)
        logger.info(f"请求 {request.request_id} 处理完成")
    
    def _result_processor_loop(self):
        """
        结果处理线程循环
        """
        while self.is_running:
            try:
                # 从结果队列获取处理完成的请求
                request = self.result_queue.get(timeout=1.0)
                
                # 这里可以添加结果后续处理逻辑
                # 例如：保存结果到数据库、发送回调等
                
                self.result_queue.task_done()
            
            except queue.Empty:
                continue
            
            except Exception as e:
                logger.error(f"结果处理线程出错: {str(e)}", exc_info=True)
    
    def _update_stats(self, response_time: float):
        """
        更新统计信息
        
        Args:
            response_time (float): 响应时间
        """
        self.total_requests += 1
        self.total_response_time += response_time
    
    def enqueue_request(self, protein_a: str, protein_b: str) -> str:
        """
        将推理请求加入队列
        
        Args:
            protein_a (str): 蛋白质A的序列
            protein_b (str): 蛋白质B的序列
            
        Returns:
            str: 请求ID
        """
        if not self.is_running:
            raise RuntimeError("推理服务器未运行")
        
        # 生成请求ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(time.time() * 1000)}"
        
        # 创建请求对象
        request = InferenceRequest(
            request_id=request_id,
            protein_a=protein_a,
            protein_b=protein_b,
            request_time=time.time()
        )
        
        # 加入请求队列
        self.request_queue.put(request)
        
        logger.info(f"请求 {request_id} 已加入队列")
        return request_id
    
    def batch_enqueue_requests(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """
        批量将推理请求加入队列
        
        Args:
            pairs (List[Tuple[str, str]]): 蛋白质对列表
            
        Returns:
            List[str]: 请求ID列表
        """
        if not self.is_running:
            raise RuntimeError("推理服务器未运行")
        
        request_ids = []
        
        for protein_a, protein_b in pairs:
            request_id = self.enqueue_request(protein_a, protein_b)
            request_ids.append(request_id)
        
        return request_ids
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取服务器状态
        
        Returns:
            Dict[str, Any]: 服务器状态信息
        """
        # 计算平均响应时间
        avg_response_time = self.total_response_time / self.total_requests if self.total_requests > 0 else 0.0
        
        return {
            'is_running': self.is_running,
            'device': self.device,
            'model_name': ((self.config.get('model', {}) or {}).get('gnn_ppi', {}) or {}).get('model_name', 'unknown'),
            'num_workers': self.num_workers,
            'queue_size': self.request_queue.qsize(),
            'total_requests': self.total_requests,
            'average_response_time': avg_response_time,
            'current_requests': self.request_counter
        }
    
    def process_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        批量处理推理请求
        
        Args:
            pairs (List[Tuple[str, str]]): 蛋白质对列表
            
        Returns:
            List[Dict[str, Any]]: 推理结果列表
        """
        logger.info(f"开始批量处理 {len(pairs)} 个请求")
        start_time = time.time()

        results = []
        for protein_a, protein_b in pairs:
            req = InferenceRequest(
                request_id=f"batch_{int(time.time() * 1000)}",
                protein_a=protein_a,
                protein_b=protein_b,
                request_time=time.time(),
            )
            self._process_request(req)
            results.append(req.response)

        logger.info(f"批量处理完成，耗时: {time.time() - start_time:.2f}秒")
        return results

    def _sequence_to_id(self, sequence: str) -> str:
        seq = (sequence or "").strip().upper()
        mapped = getattr(self.graph_builder, "sequence_to_id", {}).get(seq)
        if mapped:
            return mapped
        digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]
        return f"SEQ_{digest}"

def server_main(argv: Optional[List[str]] = None):
    """
    启动推理服务器
    """
    parser = argparse.ArgumentParser(description="启动 PGPR 推理服务器")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点路径")
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=4, help="工作线程数量")
    parser.add_argument('--batch-size', type=int, default=32, help="批处理大小")
    args = parser.parse_args(argv)
    
    # 检查配置文件和模型检查点是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型检查点不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 初始化并启动推理服务器
    server = PPIInferenceServer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    try:
        server.start()
        
        print(f"推理服务器已启动 (host={args.host}, port={args.port})，按Ctrl+C停止")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        server.stop()
        print("服务器已停止")
    
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        server.stop()
        sys.exit(1)

if __name__ == "__main__":
    server_main()
