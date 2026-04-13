#!/usr/bin/env python3
"""
PGPR (Policy-Guided Path Reasoning) 主入口点

支持的命令：
  train          训练模型
  evaluate       评估模型
  predict        预测蛋白质对相互作用
  serve          启动推理服务器
  export         导出模型
  ablation       运行消融实验
  hyper          运行超参数调优
  setup          设置数据和模型

所有命令都提供详细的帮助信息，可以使用 --help 参数查看
"""

import os
import sys

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import time
import traceback

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from utils.logger import setup_logger

# 设置日志
log_path = os.path.join(project_root, 'artifacts', 'logs')
setup_logger(log_path, log_name=None, level=logging.INFO)
logger = logging.getLogger()

def setup_logging(level=logging.INFO):
    """
    设置日志级别
    
    Args:
        level: 日志级别
    """
    setup_logger(os.path.join(project_root, 'artifacts', 'logs'), log_name=None, level=level)

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='PGPR: Policy-Guided Path Reasoning for Protein-Protein Interaction Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 主命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='configs/training.yaml',
                            help='训练配置文件路径（默认: configs/training.yaml）')
    train_parser.add_argument('--model-config', type=str, default='configs/model.yaml',
                            help='模型配置文件路径（默认: configs/model.yaml）')
    train_parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                            help='数据配置文件路径（默认: configs/data.yaml）')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                            help='从检查点恢复训练')
    train_parser.add_argument('--fine-tune', action='store_true',
                            help='仅微调模型，不从头训练')
    train_parser.add_argument('--gpu', type=int, default=None, nargs='+',
                            help='使用的GPU设备ID（默认: 所有可用GPU）')
    train_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # evaluate 命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--config', type=str, default='configs/training.yaml',
                            help='配置文件路径（默认: configs/training.yaml）')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='模型检查点路径')
    eval_parser.add_argument('--data', type=str, default=None,
                            help='评估数据集路径（默认: 配置文件中指定的测试集）')
    eval_parser.add_argument('--batch-size', type=int, default=None,
                            help='评估批次大小（默认: 配置文件中指定）')
    eval_parser.add_argument('--metrics', type=str, default='f1,accuracy,auc',
                            help='评估指标，多个指标用逗号分隔（默认: f1,accuracy,auc）')
    eval_parser.add_argument('--output', type=str, default='artifacts/metrics/evaluation',
                            help='评估结果输出目录（默认: artifacts/metrics/evaluation）')
    eval_parser.add_argument('--mode', type=str, choices=['standard', 'new_protein', 'zero_shot'], default='standard',
                            help='评估模式（默认: standard）')
    eval_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='预测蛋白质对相互作用')
    predict_parser.add_argument('--config', type=str, default='configs/model.yaml',
                            help='配置文件路径（默认: configs/model.yaml）')
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                            help='模型检查点路径')
    predict_parser.add_argument('--protein-a', type=str, required=True,
                            help='蛋白质A的氨基酸序列')
    predict_parser.add_argument('--protein-b', type=str, required=True,
                            help='蛋白质B的氨基酸序列')
    predict_parser.add_argument('--output', type=str, default=None,
                            help='预测结果输出路径（默认: 标准输出）')
    predict_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # serve 命令
    serve_parser = subparsers.add_parser('serve', help='启动推理服务器')
    serve_parser.add_argument('--config', type=str, default='configs/model.yaml',
                            help='配置文件路径（默认: configs/model.yaml）')
    serve_parser.add_argument('--checkpoint', type=str, required=True,
                            help='模型检查点路径')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0',
                            help='服务器主机（默认: 0.0.0.0）')
    serve_parser.add_argument('--port', type=int, default=8000,
                            help='服务器端口（默认: 8000）')
    serve_parser.add_argument('--workers', type=int, default=4,
                            help='工作线程数（默认: 4）')
    serve_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # export 命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--config', type=str, default='configs/model.yaml',
                            help='配置文件路径（默认: configs/model.yaml）')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                            help='模型检查点路径')
    export_parser.add_argument('--output', type=str, default='export/models',
                            help='模型导出路径（默认: export/models）')
    export_parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'package'],
                            default='package', help='导出格式（默认: package）')
    export_parser.add_argument('--quantize', action='store_true',
                            help='导出量化模型')
    export_parser.add_argument('--prune', type=float, default=None,
                            help='导出剪枝模型（0.0-1.0）')
    export_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # ablation 命令
    ablation_parser = subparsers.add_parser('ablation', help='运行消融实验')
    ablation_parser.add_argument('--config', type=str, default='configs/training.yaml',
                            help='配置文件路径（默认: configs/training.yaml）')
    ablation_parser.add_argument('--output', type=str, default='artifacts/ablation_results',
                            help='实验结果输出路径（默认: artifacts/ablation_results）')
    ablation_parser.add_argument('--components', action='store_true',
                            help='运行组件消融实验')
    ablation_parser.add_argument('--steps', action='store_true',
                            help='运行探索步数消融实验')
    ablation_parser.add_argument('--all', action='store_true',
                            help='运行所有消融实验')
    ablation_parser.add_argument('--no-visualize', action='store_true',
                            help='不生成可视化图表')
    ablation_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # hyper 命令
    hyper_parser = subparsers.add_parser('hyper', help='运行超参数调优')
    hyper_parser.add_argument('--config', type=str, default='configs/training.yaml',
                            help='配置文件路径（默认: configs/training.yaml）')
    hyper_parser.add_argument('--output', type=str, default='artifacts/hyper_results',
                            help='调优结果输出路径（默认: artifacts/hyper_results）')
    hyper_parser.add_argument('--n-trials', type=int, default=50,
                            help='调优试验次数（默认: 50）')
    hyper_parser.add_argument('--timeout', type=int, default=None,
                            help='调优超时时间（秒）')
    hyper_parser.add_argument('--resume', action='store_true',
                            help='从上次调优结果恢复')
    hyper_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # setup 命令
    setup_parser = subparsers.add_parser('setup', help='设置数据和模型')
    setup_parser.add_argument('--data', action='store_true',
                            help='设置数据')
    setup_parser.add_argument('--models', action='store_true',
                            help='设置模型')
    setup_parser.add_argument('--all', action='store_true',
                            help='设置数据和模型')
    setup_parser.add_argument('--large', action='store_true',
                            help='下载大型模型（ESM-2 3B等）')
    setup_parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    
    # 添加通用参数
    for p in [train_parser, eval_parser, predict_parser, serve_parser, export_parser, ablation_parser, hyper_parser]:
        p.add_argument('--config-override', type=str, nargs='*',
                      help='覆盖配置文件中的参数，格式：section.key=value')
    
    # 版本信息
    parser.add_argument('--version', action='version', version='PGPR v1.0.0')
    
    args = parser.parse_args()
    
    # 验证命令
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 验证setup命令
    if args.command == 'setup' and not any([args.data, args.models, args.all]):
        setup_parser.print_help()
        print("\n错误: 必须指定 --data, --models 或 --all 参数")
        sys.exit(1)
    
    # 验证ablation命令
    if args.command == 'ablation' and not any([args.components, args.steps, args.all]):
        ablation_parser.print_help()
        print("\n错误: 必须指定 --components, --steps 或 --all 参数")
        sys.exit(1)
    
    return args

def handle_train(args):
    """
    处理train命令
    """
    logger.info("开始训练模型")
    
    from experiments.run_training import train_main
    
    # 构建命令行参数
    train_args = [
        '--config', args.config,
        '--model-config', args.model_config,
        '--data-config', args.data_config
    ]
    
    if args.checkpoint:
        train_args.extend(['--checkpoint', args.checkpoint])
    
    if args.fine_tune:
        train_args.append('--fine-tune')
    
    if args.gpu:
        train_args.extend(['--gpu'] + [str(g) for g in args.gpu])
    
    if args.verbose:
        train_args.append('--verbose')
    
    if args.config_override:
        train_args.extend(['--config-override'] + args.config_override)
    
    # 调用训练函数
    try:
        train_main(train_args)
        logger.info("训练完成，准备启动自动评估...")
        
        best_candidates = []
        for d in [os.path.join('artifacts', 'checkpoints'), 'checkpoints', os.path.join('output', 'checkpoints')]:
            p = os.path.join(d, 'best_model.pth')
            if os.path.exists(p):
                best_candidates.append(p)
            if os.path.isdir(d):
                for root, _, files in os.walk(d):
                    if 'best_model.pth' in files:
                        best_candidates.append(os.path.join(root, 'best_model.pth'))
        best_checkpoint = None
        if best_candidates:
            best_checkpoint = max(best_candidates, key=lambda p: os.path.getmtime(p))
        
        if best_checkpoint and os.path.exists(best_checkpoint):
            logger.info(f"发现最佳模型检查点: {best_checkpoint}，开始评估并对比 LLAPA...")
            from experiments.run_evaluation import evaluate_main
            output_dir = os.path.join('artifacts', 'metrics', 'evaluation')
            try:
                cm = ConfigManager()
                cfgs = [
                    os.path.join(project_root, 'configs', 'base.yaml'),
                    os.path.join(project_root, 'configs', 'model.yaml'),
                    os.path.join(project_root, 'configs', 'training.yaml'),
                    os.path.join(project_root, 'configs', 'data.yaml'),
                    args.config,
                ]
                if getattr(args, "model_config", None):
                    cfgs.append(args.model_config)
                if getattr(args, "data_config", None):
                    cfgs.append(args.data_config)
                cm.load_multiple_configs(cfgs)
                results_dir = (cm.config.get('paths', {}) or {}).get('results_dir')
                if results_dir:
                    output_dir = os.path.join(results_dir, 'evaluation')
            except Exception:
                pass
            eval_args = [
                '--config', args.config,
                '--checkpoint', best_checkpoint,
                '--metrics', 'f1,accuracy,auc',
                '--output-dir', output_dir,
                '--mode', 'standard'
            ]
            if args.verbose:
                eval_args.append('--verbose')
            
            evaluate_main(eval_args)
            logger.info("自动评估与 LLAPA 对比任务已圆满完成！")
        else:
            logger.warning(f"未找到最佳模型检查点 {best_checkpoint}，跳过自动评估。")
            
    except Exception as e:
        logger.error(f"训练或自动评估过程中发生错误: {e}")
        traceback.print_exc()

def handle_evaluate(args):
    """
    处理evaluate命令
    """
    logger.info("开始评估模型")
    
    from experiments.run_evaluation import evaluate_main
    
    # 构建命令行参数
    eval_args = [
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--output-dir', args.output,
        '--mode', args.mode
    ]
    
    if args.metrics:
        eval_args.extend(['--metrics', args.metrics])
        
    if args.data:
        eval_args.extend(['--data-file', args.data])
    
    if args.batch_size:
        eval_args.extend(['--batch-size', str(args.batch_size)])
    
    if args.verbose:
        eval_args.append('--verbose')
    
    if args.config_override:
        eval_args.extend(['--config-override'] + args.config_override)
    
    # 调用评估函数
    evaluate_main(eval_args)

def handle_predict(args):
    """
    处理predict命令
    """
    logger.info("开始预测蛋白质对相互作用")
    
    # 导入预测相关模块
    from deployment.inference_server import PPIInferenceServer
    
    # 加载配置
    config_manager = ConfigManager()
    config_manager.load_config(args.config)
    
    # 初始化推理服务器（同步推理，不启动后台线程）
    server = PPIInferenceServer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_workers=1,
        batch_size=1
    )
    
    try:
        responses = server.process_batch([(args.protein_a, args.protein_b)])
        result = responses[0] if responses else {"status": "error", "message": "empty_response"}
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"预测结果已保存到: {args.output}")
        else:
            print("\n预测结果:")
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        print(f"\n预测失败: {str(e)}")
        sys.exit(1)

def handle_serve(args):
    """
    处理serve命令
    """
    logger.info("开始启动推理服务器")
    
    from deployment.inference_server import server_main
    
    # 构建命令行参数
    serve_args = [
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--host', args.host,
        '--port', str(args.port),
        '--workers', str(args.workers)
    ]
    
    if args.verbose:
        serve_args.append('--verbose')
    
    if args.config_override:
        serve_args.extend(['--config-override'] + args.config_override)
    
    # 调用服务器启动函数
    server_main(serve_args)

def handle_export(args):
    """
    处理export命令
    """
    logger.info("开始导出模型")
    
    from deployment.model_exporter import export_main
    
    # 构建命令行参数
    export_args = [
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--output', args.output
    ]
    
    if args.format == 'onnx':
        export_args.append('--onnx')
    elif args.format == 'torchscript':
        export_args.append('--torchscript')
    elif args.format == 'package':
        export_args.append('--package')
    
    if args.quantize:
        export_args.append('--quantize')
    
    if args.prune:
        export_args.extend(['--prune', str(args.prune)])
    
    if args.verbose:
        export_args.append('--verbose')
    
    # 调用导出函数
    export_main(export_args)

def handle_ablation(args):
    """
    处理ablation命令
    """
    logger.info("开始运行消融实验")
    
    from experiments.ablation_study import ablation_main
    
    # 构建命令行参数
    ablation_args = [
        '--config', args.config,
        '--output', args.output
    ]
    
    if args.components:
        ablation_args.append('--components')
    
    if args.steps:
        ablation_args.append('--steps')
    
    if args.all:
        ablation_args.append('--all')
    
    if args.no_visualize:
        ablation_args.append('--no-visualize')
    
    if args.verbose:
        ablation_args.append('--verbose')
    
    if args.config_override:
        ablation_args.extend(['--config-override'] + args.config_override)
    
    # 调用消融实验函数
    ablation_main(ablation_args)

def handle_hyper(args):
    """
    处理hyper命令
    """
    logger.info("开始运行超参数调优")
    
    from experiments.hyperparameter_tuning import hyper_main
    
    # 构建命令行参数
    hyper_args = [
        '--config', args.config,
        '--output', args.output,
        '--n-trials', str(args.n_trials)
    ]
    
    if args.timeout:
        hyper_args.extend(['--timeout', str(args.timeout)])
    
    if args.resume:
        hyper_args.append('--resume')
    
    if args.verbose:
        hyper_args.append('--verbose')
    
    if args.config_override:
        hyper_args.extend(['--config-override'] + args.config_override)
    
    # 调用超参数调优函数
    hyper_main(hyper_args)

def handle_setup(args):
    """
    处理setup命令
    """
    logger.info("开始设置数据和模型")
    
    # 执行setup脚本
    setup_scripts = []
    
    if args.data or args.all:
        setup_scripts.append('./scripts/setup_data.sh')
    
    if args.models or args.all:
        setup_scripts.append('./scripts/download_models.sh')
        if args.large:
            setup_scripts[-1] += ' --large'
    
    for script in setup_scripts:
        logger.info(f"执行脚本: {script}")
        
        # 检查脚本是否存在
        if not os.path.exists(script):
            logger.error(f"脚本不存在: {script}")
            print(f"错误: 脚本不存在: {script}")
            sys.exit(1)
        
        # 添加执行权限
        os.chmod(script, 0o755)
        
        # 执行脚本
        try:
            result = os.system(script)
            if result != 0:
                logger.error(f"脚本执行失败: {script}")
                print(f"错误: 脚本执行失败: {script}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"执行脚本时发生错误: {str(e)}", exc_info=True)
            print(f"错误: 执行脚本时发生错误: {str(e)}")
            sys.exit(1)
    
    logger.info("设置完成")
    print("\n设置完成！")

def main():
    """
    主函数
    """
    start_time = time.time()
    
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置日志级别
        if args.verbose:
            setup_logging(logging.DEBUG)
            logger.info("详细日志已开启")
        
        # 处理不同命令
        if args.command == 'train':
            handle_train(args)
        elif args.command == 'evaluate':
            handle_evaluate(args)
        elif args.command == 'predict':
            handle_predict(args)
        elif args.command == 'serve':
            handle_serve(args)
        elif args.command == 'export':
            handle_export(args)
        elif args.command == 'ablation':
            handle_ablation(args)
        elif args.command == 'hyper':
            handle_hyper(args)
        elif args.command == 'setup':
            handle_setup(args)
        else:
            print(f"未知命令: {args.command}")
            sys.exit(1)
            
        # 计算执行时间
        execution_time = time.time() - start_time
        logger.info(f"命令执行完成，耗时: {execution_time:.2f}秒")
        
    except KeyboardInterrupt:
        logger.info("收到键盘中断，退出程序")
        print("\n程序已退出")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        print(f"\n程序执行失败: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
