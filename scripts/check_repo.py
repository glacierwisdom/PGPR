#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查GitHub仓库是否存在的脚本
"""

import requests
import sys

def check_github_repo(repo_url):
    """
    检查GitHub仓库是否存在
    
    Args:
        repo_url: GitHub仓库URL
        
    Returns:
        tuple: (存在状态, 仓库信息)
    """
    # 转换为API URL
    if repo_url.endswith('/'):
        repo_url = repo_url[:-1]
    
    parts = repo_url.split('/')
    if len(parts) < 2:
        return False, "无效的GitHub仓库URL"
    
    owner = parts[-2]
    repo = parts[-1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            repo_info = response.json()
            return True, f"仓库存在: {repo_info['full_name']}\n描述: {repo_info.get('description', '无描述')}\n星标数: {repo_info['stargazers_count']}\n分支数: {repo_info['forks_count']}"
        elif response.status_code == 404:
            return False, "仓库不存在或已被删除"
        elif response.status_code == 403:
            return False, "API请求限制，请稍后再试"
        else:
            return False, f"请求失败，状态码: {response.status_code}"
    except Exception as e:
        return False, f"请求出错: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_repo.py <GitHub仓库URL>")
        print("示例: python check_repo.py https://github.com/lijfrank-open/JmcPPI")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    print(f"检查仓库: {repo_url}")
    print("=" * 60)
    
    exists, message = check_github_repo(repo_url)
    print(message)
    
    if exists:
        print("\n仓库可访问！")
        sys.exit(0)
    else:
        print("\n仓库不可访问！")
        sys.exit(1)
