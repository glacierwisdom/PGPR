import yaml
import os
import argparse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理器，负责加载、合并和验证配置文件
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            base_config_path (str, optional): 基础配置文件路径
        """
        self.config: Dict[str, Any] = {}
        
        if base_config_path:
            self.load_config(base_config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载单个配置文件
        
        Args:
            config_path (str): 配置文件路径
            
        Returns:
            Dict[str, Any]: 加载的配置
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    
    def merge_config(self, config_dict: Dict[str, Any]) -> None:
        """
        合并配置到当前配置
        
        Args:
            config_dict (Dict[str, Any]): 要合并的配置
        """
        self._deep_merge(self.config, config_dict)
        logger.info("配置合并完成")
    
    def _deep_merge(self, dest: Dict[str, Any], src: Dict[str, Any]) -> None:
        """
        深度合并两个字典
        
        Args:
            dest (Dict[str, Any]): 目标字典
            src (Dict[str, Any]): 源字典
        """
        for key, value in src.items():
            if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
                self._deep_merge(dest[key], value)
            else:
                dest[key] = value
    
    def load_multiple_configs(self, config_paths: list) -> None:
        """
        加载并合并多个配置文件
        
        Args:
            config_paths (list): 配置文件路径列表
        """
        for config_path in config_paths:
            config = self.load_config(config_path)
            self.merge_config(config)
    
    def parse_cli_args(self, args: argparse.Namespace) -> None:
        """
        解析命令行参数并覆盖配置
        
        Args:
            args (argparse.Namespace): 命令行参数
        """
        arg_dict = vars(args)
        
        # 过滤掉为None的参数
        filtered_args = {k: v for k, v in arg_dict.items() if v is not None}
        
        if filtered_args:
            logger.info(f"使用命令行参数覆盖配置: {filtered_args}")
            self.merge_config(filtered_args)
    
    def validate_config(self, required_keys: list) -> bool:
        """
        验证配置完整性
        
        Args:
            required_keys (list): 必需的配置键列表
            
        Returns:
            bool: 配置是否完整
        """
        missing_keys = []
        
        for key in required_keys:
            if not self._has_key(self.config, key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"配置缺少必需的键: {missing_keys}")
            return False
        
        logger.info("配置验证通过")
        return True

    def override_config(self, overrides: list) -> None:
        override_config: Dict[str, Any] = {}
        for override in overrides or []:
            if '=' not in override:
                continue
            key, value = override.split('=', 1)
            keys = key.split('.')
            current: Dict[str, Any] = override_config
            for k in keys[:-1]:
                if k not in current or not isinstance(current.get(k), dict):
                    current[k] = {}
                current = current[k]

            if isinstance(value, str):
                lower = value.lower()
                if lower in ("true", "false"):
                    value = lower == "true"
                elif lower in ("none", "null"):
                    value = None
                else:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass

            current[keys[-1]] = value

        if override_config:
            self.merge_config(override_config)
    
    def _has_key(self, config: Dict[str, Any], key: str) -> bool:
        """
        检查配置中是否存在指定的键（支持嵌套键，如 'model.hidden_size'）
        
        Args:
            config (Dict[str, Any]): 配置字典
            key (str): 要检查的键（支持嵌套键）
            
        Returns:
            bool: 键是否存在
        """
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键）
        
        Args:
            key (str): 配置键（支持嵌套键，如 'model.hidden_size'）
            default (Any, optional): 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def save_config(self, output_path: str) -> None:
        """
        保存当前配置到文件
        
        Args:
            output_path (str): 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已保存到: {output_path}")
    
    def __getitem__(self, key: str) -> Any:
        """
        支持使用字典语法获取配置值
        
        Args:
            key (str): 配置键
            
        Returns:
            Any: 配置值
        """
        return self.get(key)
    
    def __repr__(self) -> str:
        """
        返回配置的字符串表示
        
        Returns:
            str: 配置的字符串表示
        """
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
