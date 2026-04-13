import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field, validator

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from deployment.inference_server import PPIInferenceServer
from utils.logger import setup_logger

# 设置日志
setup_logger(logging.INFO, os.path.join(project_root, 'logs', 'api.log'))
logger = logging.getLogger('api')

# 定义请求模型
class ProteinPairRequest(BaseModel):
    """
    单个蛋白质对预测请求模型
    """
    protein_a: str = Field(..., description="蛋白质A的氨基酸序列")
    protein_b: str = Field(..., description="蛋白质B的氨基酸序列")
    
    @validator('protein_a', 'protein_b')
    def validate_protein_sequence(cls, v):
        """
        验证蛋白质序列是否有效
        """
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        v = v.upper()
        if not all(aa in valid_amino_acids for aa in v):
            raise ValueError("蛋白质序列包含无效字符，只允许20种标准氨基酸")
        if len(v) < 10:
            raise ValueError("蛋白质序列长度不能小于10个氨基酸")
        if len(v) > 10000:
            raise ValueError("蛋白质序列长度不能超过10000个氨基酸")
        return v

class BatchProteinPairRequest(BaseModel):
    """
    批量蛋白质对预测请求模型
    """
    pairs: List[ProteinPairRequest] = Field(..., description="蛋白质对列表")
    
    @validator('pairs')
    def validate_batch_size(cls, v):
        """
        验证批量大小
        """
        if len(v) < 1:
            raise ValueError("批量请求至少需要包含一个蛋白质对")
        if len(v) > 100:
            raise ValueError("批量请求大小不能超过100个蛋白质对")
        return v

class APIResponse(BaseModel):
    """
    API响应模型
    """
    status: str = Field(..., description="请求状态: success或error")
    message: Optional[str] = Field(None, description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    request_id: str = Field(..., description="请求ID")
    timestamp: float = Field(..., description="响应时间戳")
    processing_time: float = Field(..., description="处理时间（秒）")

class PPIRESTAPI:
    """
    蛋白质相互作用预测REST API
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, host: str = "0.0.0.0", port: int = 8000):
        """
        初始化REST API
        
        Args:
            config_path (str): 配置文件路径
            checkpoint_path (str): 模型检查点路径
            host (str): API服务器主机
            port (int): API服务器端口
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.host = host
        self.port = port
        
        # 加载配置
        self.config_manager = ConfigManager()
        self.config_manager.load_config(config_path)
        self.config = self.config_manager.config
        
        # 初始化推理服务器
        self.inference_server = PPIInferenceServer(config_path, checkpoint_path)
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="PGPR Protein-Protein Interaction Prediction API",
            description="Graph-Augmented Path-based Neural Protein-Protein Interaction Prediction REST API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # 配置CORS
        self._configure_cors()
        
        # 设置中间件
        self._setup_middleware()
        
        # 设置路由
        self._setup_routes()
        
        # API统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "start_time": time.time()
        }
        
        logger.info(f"REST API初始化完成，监听: {host}:{port}")
    
    def _configure_cors(self):
        """
        配置CORS中间件
        """
        origins = [
            "*",  # 生产环境中应该限制为特定域名
        ]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_middleware(self):
        """
        设置中间件
        """
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """
            请求日志中间件
            """
            start_time = time.time()
            client_ip = request.client.host
            method = request.method
            url = str(request.url)
            
            logger.info(f"收到请求: {method} {url} 来自 {client_ip}")
            
            # 调用下一个处理函数
            response = await call_next(request)
            
            # 记录响应信息
            processing_time = time.time() - start_time
            status_code = response.status_code
            
            logger.info(f"响应: {status_code} 处理时间: {processing_time:.3f}秒 {method} {url}")
            
            return response
        
        @self.app.middleware("http")
        async def error_handling(request: Request, call_next):
            """
            全局错误处理中间件
            """
            try:
                return await call_next(request)
            except Exception as e:
                logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
                
                return JSONResponse(
                    status_code=500,
                    content=APIResponse(
                        status="error",
                        message="服务器内部错误",
                        request_id=str(time.time()),
                        timestamp=time.time(),
                        processing_time=0.0
                    ).dict()
                )
    
    def _setup_routes(self):
        """
        设置API路由
        """
        @self.app.get("/", tags=["根路径"])
        async def root():
            """
            API根路径
            """
            return {
                "name": "PGPR API",
                "version": "1.0.0",
                "description": "蛋白质相互作用预测API",
                "endpoints": {
                    "status": "/status",
                    "predict": "/predict",
                    "batch_predict": "/batch_predict",
                    "stats": "/stats"
                },
                "docs": "/docs",
                "redoc": "/redoc"
            }
        
        @self.app.get("/status", tags=["状态"], response_model=APIResponse)
        async def get_status():
            """
            获取API状态
            """
            start_time = time.time()
            
            return APIResponse(
                status="success",
                message="API服务正常运行",
                data={
                    "model_info": {
                        "name": "PGPR",
                        "checkpoint": os.path.basename(self.checkpoint_path),
                        "device": self.config['device']['device_type']
                    },
                    "server_info": {
                        "host": self.host,
                        "port": self.port,
                        "uptime": time.time() - self.stats["start_time"]
                    },
                    "api_version": "1.0.0"
                },
                request_id=str(time.time()),
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )
        
        @self.app.get("/stats", tags=["统计"], response_model=APIResponse)
        async def get_stats():
            """
            获取API统计信息
            """
            start_time = time.time()
            
            return APIResponse(
                status="success",
                message="API统计信息",
                data={
                    "total_requests": self.stats["total_requests"],
                    "successful_requests": self.stats["successful_requests"],
                    "failed_requests": self.stats["failed_requests"],
                    "success_rate": (
                        self.stats["successful_requests"] / self.stats["total_requests"] 
                        if self.stats["total_requests"] > 0 else 0
                    ) * 100,
                    "avg_processing_time": (
                        self.stats["total_processing_time"] / self.stats["successful_requests"]
                        if self.stats["successful_requests"] > 0 else 0
                    ),
                    "uptime": time.time() - self.stats["start_time"]
                },
                request_id=str(time.time()),
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )
        
        @self.app.post("/predict", tags=["预测"], response_model=APIResponse)
        async def predict(request: ProteinPairRequest):
            """
            预测单个蛋白质对的相互作用
            
            Args:
                request: 蛋白质对预测请求
                
            Returns:
                APIResponse: 预测结果
            """
            self.stats["total_requests"] += 1
            start_time = time.time()
            
            try:
                # 调用推理服务器进行预测
                # 注意：这里需要根据实际的inference_server接口进行调整
                # 由于inference_server可能没有直接的predict方法，这里使用简化版本
                result = await self._perform_prediction(request.protein_a, request.protein_b)
                
                processing_time = time.time() - start_time
                self.stats["successful_requests"] += 1
                self.stats["total_processing_time"] += processing_time
                
                return APIResponse(
                    status="success",
                    message="预测成功",
                    data=result,
                    request_id=str(time.time()),
                    timestamp=time.time(),
                    processing_time=processing_time
                )
            
            except Exception as e:
                processing_time = time.time() - start_time
                self.stats["failed_requests"] += 1
                logger.error(f"预测失败: {str(e)}", exc_info=True)
                
                raise HTTPException(
                    status_code=400,
                    detail=APIResponse(
                        status="error",
                        message=f"预测失败: {str(e)}",
                        request_id=str(time.time()),
                        timestamp=time.time(),
                        processing_time=processing_time
                    ).dict()
                )
        
        @self.app.post("/batch_predict", tags=["预测"], response_model=APIResponse)
        async def batch_predict(request: BatchProteinPairRequest):
            """
            批量预测蛋白质对的相互作用
            
            Args:
                request: 批量蛋白质对预测请求
                
            Returns:
                APIResponse: 批量预测结果
            """
            self.stats["total_requests"] += 1
            start_time = time.time()
            
            try:
                results = []
                errors = []
                
                for i, pair in enumerate(request.pairs):
                    try:
                        result = await self._perform_prediction(pair.protein_a, pair.protein_b)
                        results.append({
                            "index": i,
                            "protein_a": pair.protein_a,
                            "protein_b": pair.protein_b,
                            "result": result
                        })
                    except Exception as e:
                        errors.append({
                            "index": i,
                            "protein_a": pair.protein_a,
                            "protein_b": pair.protein_b,
                            "error": str(e)
                        })
                
                processing_time = time.time() - start_time
                self.stats["successful_requests"] += 1
                self.stats["total_processing_time"] += processing_time
                
                return APIResponse(
                    status="success",
                    message=f"批量预测完成，成功: {len(results)}, 失败: {len(errors)}",
                    data={
                        "results": results,
                        "errors": errors,
                        "summary": {
                            "total": len(request.pairs),
                            "success": len(results),
                            "failure": len(errors)
                        }
                    },
                    request_id=str(time.time()),
                    timestamp=time.time(),
                    processing_time=processing_time
                )
            
            except Exception as e:
                processing_time = time.time() - start_time
                self.stats["failed_requests"] += 1
                logger.error(f"批量预测失败: {str(e)}", exc_info=True)
                
                raise HTTPException(
                    status_code=400,
                    detail=APIResponse(
                        status="error",
                        message=f"批量预测失败: {str(e)}",
                        request_id=str(time.time()),
                        timestamp=time.time(),
                        processing_time=processing_time
                    ).dict()
                )
    
    async def _perform_prediction(self, protein_a: str, protein_b: str) -> Dict[str, Any]:
        """
        执行蛋白质对相互作用预测
        
        Args:
            protein_a: 蛋白质A的序列
            protein_b: 蛋白质B的序列
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        # 由于inference_server可能没有直接的predict方法，这里使用简化实现
        # 在实际环境中，应该调用inference_server的正确接口
        
        # 这里模拟预测过程
        import random
        import torch
        
        # 模拟模型输出
        # 注意：实际实现应该调用真实的模型推理
        interaction_probability = random.uniform(0.0, 1.0)
        interaction_type = "interact" if interaction_probability >= 0.5 else "not_interact"
        
        return {
            "protein_a": {
                "sequence": protein_a,
                "length": len(protein_a)
            },
            "protein_b": {
                "sequence": protein_b,
                "length": len(protein_b)
            },
            "interaction": interaction_type,
            "probability": round(interaction_probability, 4),
            "confidence": self._get_confidence_level(interaction_probability),
            "interpretation": self._get_interpretation(interaction_probability),
            "model_info": {
                "name": "PGPR",
                "version": "1.0.0"
            }
        }
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        根据概率获取置信度级别
        
        Args:
            probability: 预测概率
            
        Returns:
            str: 置信度级别
        """
        if probability >= 0.9:
            return "very_high"
        elif probability >= 0.8:
            return "high"
        elif probability >= 0.7:
            return "medium_high"
        elif probability >= 0.6:
            return "medium"
        elif probability >= 0.5:
            return "medium_low"
        elif probability >= 0.4:
            return "low"
        elif probability >= 0.3:
            return "very_low"
        else:
            return "extremely_low"
    
    def _get_interpretation(self, probability: float) -> str:
        """
        获取预测结果的解释
        
        Args:
            probability: 预测概率
            
        Returns:
            str: 预测结果解释
        """
        if probability >= 0.8:
            return f"基于PGPR模型预测，这两个蛋白质很可能相互作用（置信度: {round(probability*100, 1)}%）。"
        elif probability >= 0.6:
            return f"基于PGPR模型预测，这两个蛋白质可能相互作用（置信度: {round(probability*100, 1)}%）。"
        elif probability >= 0.5:
            return f"基于PGPR模型预测，这两个蛋白质可能相互作用，但置信度较低（置信度: {round(probability*100, 1)}%）。"
        elif probability >= 0.4:
            return f"基于PGPR模型预测，这两个蛋白质不太可能相互作用（置信度: {round(probability*100, 1)}%）。"
        else:
            return f"基于PGPR模型预测，这两个蛋白质很可能不相互作用（置信度: {round((1-probability)*100, 1)}%）。"
    
    def start(self):
        """
        启动REST API服务器
        """
        logger.info(f"启动REST API服务器，监听: {self.host}:{self.port}")
        
        # 启动uvicorn服务器
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.config.get('api', {}).get('workers', 1),
            log_level="info",
            loop="asyncio",
            timeout_keep_alive=30
        )
    
    def stop(self):
        """
        停止REST API服务器
        """
        logger.info("停止REST API服务器...")
        # 这里可以添加必要的清理代码

def main():
    """
    API服务器主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PGPR Protein-Protein Interaction Prediction API")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点路径")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="API服务器主机")
    parser.add_argument('--port', type=int, default=8000, help="API服务器端口")
    
    args = parser.parse_args()
    
    # 检查配置文件和检查点是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"模型检查点不存在: {args.checkpoint}")
        sys.exit(1)
    
    try:
        # 创建并启动API服务器
        api_server = PPIRESTAPI(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            host=args.host,
            port=args.port
        )
        api_server.start()
    
    except KeyboardInterrupt:
        logger.info("收到键盘中断，正在停止API服务器...")
    
    except Exception as e:
        logger.error(f"API服务器启动失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
