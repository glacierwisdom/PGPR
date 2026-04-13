import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CacheEntry:
    """缓存条目的数据结构"""
    data: Any
    timestamp: float
    expires_at: Optional[float] = None


class Cache:
    """通用缓存基类"""
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        初始化缓存
        
        Args:
            max_size: 缓存最大条目数
            ttl: 缓存条目过期时间（秒），None表示永不过期
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
    
    def _get_key(self, data: Any) -> str:
        """生成数据的哈希键"""
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        elif isinstance(data, torch.Tensor):
            return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
        else:
            return hashlib.sha256(pickle.dumps(data)).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存条目是否过期"""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at
    
    def _evict_oldest(self) -> None:
        """移除最旧的缓存条目"""
        if not self._cache:
            return
        oldest_key = min(self._cache.items(), key=lambda x: x[1].timestamp)[0]
        del self._cache[oldest_key]
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        
        return entry.data
    
    def set(self, key: str, data: Any) -> None:
        """设置缓存数据"""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        expires_at = time.time() + self.ttl if self.ttl is not None else None
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            expires_at=expires_at
        )
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)


class MemoryCache(Cache):
    """内存缓存实现"""
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        super().__init__(max_size, ttl)


class DiskCache(Cache):
    """磁盘缓存实现"""
    def __init__(self,
                 cache_dir: Union[str, Path] = ".cache",
                 max_size: int = 1000,
                 ttl: Optional[float] = None):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_size: 缓存最大条目数
            ttl: 缓存条目过期时间（秒），None表示永不过期
        """
        super().__init__(max_size, ttl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()
    
    def _get_file_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"
    
    def _load_from_disk(self) -> None:
        """从磁盘加载缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)
                if not self._is_expired(entry):
                    key = cache_file.stem
                    self._cache[key] = entry
            except Exception as e:
                print(f"Failed to load cache file {cache_file}: {e}")
    
    def set(self, key: str, data: Any) -> None:
        """设置缓存数据（同时保存到磁盘）"""
        super().set(key, data)
        
        # 保存到磁盘
        try:
            file_path = self._get_file_path(key)
            with open(file_path, "wb") as f:
                pickle.dump(self._cache[key], f)
        except Exception as e:
            print(f"Failed to save cache to disk: {e}")
    
    def clear(self) -> None:
        """清空缓存（同时删除磁盘文件）"""
        super().clear()
        
        # 删除磁盘文件
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Failed to delete cache file {cache_file}: {e}")


class ESMEncodingCache:
    """ESM编码缓存"""
    def __init__(self,
                 cache_dir: Union[str, Path] = ".cache/esm_encodings",
                 max_size: int = 10000,
                 ttl: Optional[float] = None):
        """
        初始化ESM编码缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_size: 缓存最大条目数
            ttl: 缓存条目过期时间（秒），None表示永不过期
        """
        self.cache = DiskCache(cache_dir, max_size, ttl)
    
    def get_encoding(self, sequence: str) -> Optional[torch.Tensor]:
        """获取序列的ESM编码"""
        key = self.cache._get_key(sequence)
        return self.cache.get(key)
    
    def cache_encoding(self, sequence: str, encoding: torch.Tensor) -> None:
        """缓存序列的ESM编码"""
        key = self.cache._get_key(sequence)
        self.cache.set(key, encoding)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()


class SimilaritySearchCache:
    """相似性搜索结果缓存"""
    def __init__(self,
                 cache_dir: Union[str, Path] = ".cache/similarity_results",
                 max_size: int = 1000,
                 ttl: Optional[float] = None):
        """
        初始化相似性搜索结果缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_size: 缓存最大条目数
            ttl: 缓存条目过期时间（秒），None表示永不过期
        """
        self.cache = DiskCache(cache_dir, max_size, ttl)
    
    def get_results(self, query_features: torch.Tensor, top_k: int) -> Optional[Dict[str, Any]]:
        """获取相似性搜索结果"""
        cache_key = f"{self.cache._get_key(query_features)}_{top_k}"
        return self.cache.get(cache_key)
    
    def cache_results(self, query_features: torch.Tensor, top_k: int, results: Dict[str, Any]) -> None:
        """缓存相似性搜索结果"""
        cache_key = f"{self.cache._get_key(query_features)}_{top_k}"
        self.cache.set(cache_key, results)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()


class GraphStructureCache:
    """图结构缓存"""
    def __init__(self,
                 cache_dir: Union[str, Path] = ".cache/graph_structures",
                 max_size: int = 500,
                 ttl: Optional[float] = None):
        """
        初始化图结构缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_size: 缓存最大条目数
            ttl: 缓存条目过期时间（秒），None表示永不过期
        """
        self.cache = DiskCache(cache_dir, max_size, ttl)
    
    def get_graph(self, protein_ids: Union[str, list]) -> Optional[Dict[str, Any]]:
        """获取图结构"""
        if isinstance(protein_ids, list):
            protein_ids = "_&".join(sorted(protein_ids))
        cache_key = self.cache._get_key(protein_ids)
        return self.cache.get(cache_key)
    
    def cache_graph(self, protein_ids: Union[str, list], graph: Dict[str, Any]) -> None:
        """缓存图结构"""
        if isinstance(protein_ids, list):
            protein_ids = "_&".join(sorted(protein_ids))
        cache_key = self.cache._get_key(protein_ids)
        self.cache.set(cache_key, graph)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()


# 全局缓存实例
_esm_cache = None
_similarity_cache = None
_graph_cache = None


def get_esm_cache() -> ESMEncodingCache:
    """获取全局ESM编码缓存实例"""
    global _esm_cache
    if _esm_cache is None:
        _esm_cache = ESMEncodingCache()
    return _esm_cache


def get_similarity_cache() -> SimilaritySearchCache:
    """获取全局相似性搜索结果缓存实例"""
    global _similarity_cache
    if _similarity_cache is None:
        _similarity_cache = SimilaritySearchCache()
    return _similarity_cache


def get_graph_cache() -> GraphStructureCache:
    """获取全局图结构缓存实例"""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = GraphStructureCache()
    return _graph_cache


def clear_all_caches() -> None:
    """清空所有缓存"""
    get_esm_cache().clear()
    get_similarity_cache().clear()
    get_graph_cache().clear()
    print("All caches cleared successfully!")