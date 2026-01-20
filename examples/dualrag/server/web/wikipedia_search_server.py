#!/usr/bin/env python3
"""
Wikipedia Master Server - 分布式协调器

与 Firefox 沙盒版 Agent 配合使用
"""
import os
import asyncio
import logging
import time
import re
import json
import os
import multiprocessing
import aiohttp
from typing import Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from bs4 import BeautifulSoup

path = os.path.dirname(os.path.abspath(__file__))

# ================= 配置区域 =================
# Agent 节点列表
AGENT_NODES = [
    "http://127.0.0.1:8089",
]

# 全局 QPM 限制 - 因为沙盒操作更慢，适当降低
MAX_QPM = 500

# 请求超时 - 沙盒操作需要更长时间
AGENT_TIMEOUT = 120  # 秒

# 缓存文件路径
CACHE_FILE_PATH = f"{path}/wikipedia_cache.json"

# 缓存保存间隔
CACHE_SAVE_INTERVAL_REQUESTS = 200  # 每 200 条请求保存一次
CACHE_SAVE_INTERVAL_SECONDS = 300   # 或者每 300 秒保存一次
# ===========================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterServer")


# ==========================================
# 1. CPU 密集型解析函数 (保持不变)
# ==========================================

def _get_clean_text(element) -> str:
    if not element: 
        return ""
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _clean_cell_content(cell_str: str) -> str:
    cell_soup = BeautifulSoup(cell_str, 'html.parser')
    for br in cell_soup.find_all("br"): 
        br.replace_with(" <br> ")
    for li in cell_soup.find_all("li"): 
        li.insert(0, " • ")
        li.insert_after(" <br> ") 
    text = _get_clean_text(cell_soup)
    text = text.replace(" <br> ", "<br>")
    if text.endswith("<br>"): 
        text = text[:-4]
    return text


def _table_to_markdown(table_tag) -> str:
    rows = table_tag.find_all('tr')
    if not rows: 
        return ""
    caption = table_tag.find('caption')
    caption_text = f"\n**Table：{_get_clean_text(caption)}**\n" if caption else ""
    grid = {} 
    max_col = 0
    max_row = len(rows)
    occupied = set()
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            while (r_idx, c_idx) in occupied: 
                c_idx += 1
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            cell_text = _clean_cell_content(str(cell))
            for r in range(rowspan):
                for c in range(colspan):
                    occupied.add((r_idx + r, c_idx + c))
                    grid[(r_idx + r, c_idx + c)] = cell_text
            c_idx += colspan
        if c_idx > max_col: 
            max_col = c_idx
    
    md_lines = [caption_text]
    table_rows = []
    for r in range(max_row):
        current_row = []
        for c in range(max_col):
            val = grid.get((r, c), " ").replace("|", "|")
            current_row.append(val)
        table_rows.append(current_row)
    
    if not table_rows: 
        return ""
    
    header = table_rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_col) + " |")
    for row_data in table_rows[1:]:
        md_lines.append("| " + " | ".join(row_data) + " |")
    return "\n".join(md_lines)


def cpu_bound_parse(html_content: str, entity_name: str) -> Dict[str, Any]:
    """
    CPU 密集型解析函数，在进程池中运行
    """
    MAX_RESULT_LENGTH = 30000
    try:
        if not html_content:
            return {"error": "Empty HTML content"}

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 检查消歧页/搜索结果列表
        search_results_list = soup.find('ul', class_='mw-search-results')
        if search_results_list:
            output = ["No exact entity matched, there are some candidates:\n"]
            items = search_results_list.find_all('li', class_='mw-search-result')
            for idx, item in enumerate(items, 1):
                heading = item.find('div', class_='mw-search-result-heading')
                title = _get_clean_text(heading) if heading else "unknown"
                snippet = item.find('div', class_='searchresult')
                desc = _get_clean_text(snippet) if snippet else "No intro"
                output.append(f"{idx}. entity: {title}\n   introduction: {desc}\n")
            return {
                "output": "\n".join(output),
                "metadata": {"type": "search_results", "query": entity_name}
            }
        
        # 2. 检查无结果
        elif soup.find('p', class_='mw-search-nonefound'):
            return {
                "output": f"No results found for '{entity_name}'.",
                "metadata": {"type": "no_results", "query": entity_name}
            }
        
        # 3. 精确匹配 (实体页)
        else:
            title_tag = soup.find('h1', id='firstHeading')
            title = _get_clean_text(title_tag) if title_tag else "unknown item"
            output = [f"# {title}\n"]
            
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                return {"output": "No content found.", "metadata": {"type": "error"}}

            # 清理无用标签
            for selector in ['script', 'style', '.mw-editsection', '.reference', '.reflist', 
                             '.noprint', '.infobox', '.thumb', '.hatnote', '.ambox', 
                             '.navbox', '.sidebar', '.catlinks', '.mw-indicators']:
                for tag in content_div.select(selector):
                    tag.decompose()

            parser_output = content_div.find('div', class_='mw-parser-output')
            target_container = parser_output if parser_output else content_div
            target_tags = ['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'dl', 'table', 'blockquote']
            
            for element in target_container.find_all(target_tags, recursive=True):
                if element.find_parents(target_tags): 
                    continue

                if element.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(element.name.replace('h', ''))
                    text = _get_clean_text(element)
                    if text: 
                        output.append(f"\n{'#' * level} {text}\n")
                
                elif element.name == 'p':
                    text = _get_clean_text(element)
                    if text and len(text) > 1: 
                        output.append(f"{text}\n")

                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li', recursive=False):
                        output.append(f"- {_get_clean_text(li)}")
                    output.append("")
                    
                elif element.name == 'dl':
                    for dd in element.find_all('dd'):
                        output.append(f"> {_get_clean_text(dd)}\n")

                elif element.name == 'blockquote':
                    output.append(f"> {_get_clean_text(element)}\n")
                
                elif element.name == 'table':
                    if 'wikitable' in element.get('class', []) or element.find('caption'):
                        md_table = _table_to_markdown(element)
                        if md_table: 
                            output.append(f"\n{md_table}\n")

            full_text = "\n".join(output)
            return {
                "output": full_text[:MAX_RESULT_LENGTH],
                "metadata": {"type": "entity_page", "query": entity_name}
            }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 2. 分布式请求管理器
# ==========================================

class RateLimiter:
    """异步速率限制器"""
    
    def __init__(self, qpm: int):
        self.interval = 60.0 / qpm
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait_for_token(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            wait_time = self.interval - elapsed
            
            if wait_time > 0:
                logger.debug(f"Rate limit active, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class NodeHealth:
    """节点健康状态追踪"""
    
    def __init__(self, node_url: str):
        self.url = node_url
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.is_healthy = True
        self.cooldown_until = 0
    
    def record_success(self):
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.is_healthy = True
    
    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            # 连续失败3次，进入冷却期
            self.is_healthy = False
            self.cooldown_until = time.time() + 60  # 冷却60秒
    
    def is_available(self) -> bool:
        if time.time() > self.cooldown_until:
            self.is_healthy = True
        return self.is_healthy


class DistributedSearchManager:
    """分布式搜索管理器"""
    
    def __init__(self, agent_nodes: List[str], qpm: int, cache_file_path: str):
        self.nodes = [NodeHealth(url) for url in agent_nodes]
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.processing_requests: Dict[str, asyncio.Event] = {}
        
        # 缓存文件路径
        self.cache_file_path = cache_file_path
        
        # 进程池用于 CPU 密集型解析
        cpu_count = multiprocessing.cpu_count()
        self.process_executor = ProcessPoolExecutor(max_workers=min(cpu_count, 8))
        
        # 限流
        self.rate_limiter = RateLimiter(qpm)
        self.node_index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0
        }
        
        # 缓存保存相关
        self.requests_since_last_save = 0
        self.last_save_time = time.time()
        self.cache_save_lock = asyncio.Lock()
        
        # 启动时加载缓存
        self._load_cache_from_file()

    def _load_cache_from_file(self):
        """从文件加载缓存"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"成功从 {self.cache_file_path} 加载 {len(self.cache)} 条缓存记录")
            else:
                logger.info(f"缓存文件 {self.cache_file_path} 不存在，将创建新缓存")
                self.cache = {}
        except json.JSONDecodeError as e:
            logger.error(f"缓存文件 JSON 解析失败: {e}，将使用空缓存")
            self.cache = {}
        except Exception as e:
            logger.error(f"加载缓存文件失败: {e}，将使用空缓存")
            self.cache = {}

    def _save_cache_to_file_sync(self):
        """同步保存缓存到文件（供进程退出时使用）"""
        try:
            # 确保目录存在
            cache_dir = os.path.dirname(self.cache_file_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # 先写入临时文件，再原子性重命名
            temp_path = self.cache_file_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            # 原子性重命名
            os.replace(temp_path, self.cache_file_path)
            logger.info(f"成功保存 {len(self.cache)} 条缓存记录到 {self.cache_file_path}")
        except Exception as e:
            logger.error(f"保存缓存文件失败: {e}")

    async def _save_cache_to_file(self):
        """异步保存缓存到文件"""
        async with self.cache_save_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_cache_to_file_sync)
            self.requests_since_last_save = 0
            self.last_save_time = time.time()

    async def _check_and_save_cache(self):
        """检查是否需要保存缓存"""
        should_save = False
        
        # 基于请求数
        if self.requests_since_last_save >= CACHE_SAVE_INTERVAL_REQUESTS:
            should_save = True
            logger.info(f"达到 {CACHE_SAVE_INTERVAL_REQUESTS} 条请求，触发缓存保存")
        
        # 基于时间
        elif time.time() - self.last_save_time >= CACHE_SAVE_INTERVAL_SECONDS:
            should_save = True
            logger.info(f"距上次保存超过 {CACHE_SAVE_INTERVAL_SECONDS} 秒，触发缓存保存")
        
        if should_save:
            await self._save_cache_to_file()

    async def _get_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=AGENT_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    def _get_next_healthy_node(self) -> Optional[NodeHealth]:
        """获取下一个健康的节点"""
        attempts = 0
        while attempts < len(self.nodes):
            node = self.nodes[self.node_index % len(self.nodes)]
            self.node_index += 1
            if node.is_available():
                return node
            attempts += 1
        
        # 所有节点都不健康，强制返回第一个
        logger.warning("所有节点都不健康，强制使用第一个节点")
        return self.nodes[0] if self.nodes else None

    async def _fetch_from_agent(self, entity_name: str) -> Optional[str]:
        """从 Agent 获取 HTML"""
        await self.rate_limiter.wait_for_token()
        
        session = await self._get_session()
        retries = len(self.nodes)  # 最多尝试所有节点
        
        for _ in range(retries):
            node = self._get_next_healthy_node()
            if not node:
                break
                
            target_url = f"{node.url}/fetch"
            
            try:
                logger.info(f"向 {node.url} 请求: {entity_name}")
                async with session.get(target_url, params={"entity": entity_name}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        node.record_success()
                        logger.info(f"从 {data.get('node_id', 'unknown')} 成功获取: {entity_name}")
                        return data.get("html", "")
                    elif resp.status == 429:
                        logger.warning(f"Agent {node.url} 限流 (429)")
                        node.record_failure()
                        continue
                    elif resp.status == 502:
                        error_detail = await resp.text()
                        logger.warning(f"Agent {node.url} 获取失败 (502): {error_detail}")
                        node.record_failure()
                        continue
                    else:
                        logger.error(f"Agent {node.url} 返回 {resp.status}")
                        node.record_failure()
                        
            except asyncio.TimeoutError:
                logger.error(f"请求 {node.url} 超时")
                node.record_failure()
            except Exception as e:
                logger.error(f"连接 {node.url} 失败: {e}")
                node.record_failure()
        
        return None

    async def get_result(self, entity_name: str) -> Dict[str, Any]:
        """获取搜索结果"""
        self.stats['total_requests'] += 1
        self.requests_since_last_save += 1
        
        # 缓存检查
        if entity_name in self.cache:
            cached = self.cache[entity_name]
            if not cached.get("error"):
                self.stats['cache_hits'] += 1
                logger.info(f"缓存命中: {entity_name}")
                # 检查是否需要保存缓存（即使命中也计数）
                await self._check_and_save_cache()
                return cached

        # 请求合并
        if entity_name in self.processing_requests:
            logger.info(f"等待相同请求: {entity_name}")
            await self.processing_requests[entity_name].wait()
            return self.cache.get(entity_name, {"error": "Wait failed"})

        event = asyncio.Event()
        self.processing_requests[entity_name] = event

        try:
            # 1. 从 Agent 获取 HTML
            html_content = await self._fetch_from_agent(entity_name)
            
            if not html_content:
                self.stats['failed_requests'] += 1
                result = {"error": "Failed to fetch content from any agent"}
            else:
                # 2. 本地解析
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.process_executor, 
                    cpu_bound_parse,
                    html_content, 
                    entity_name
                )
                
                if result.get("error"):
                    self.stats['failed_requests'] += 1
                else:
                    self.stats['successful_requests'] += 1

            self.cache[entity_name] = result
            
            # 检查是否需要保存缓存
            await self._check_and_save_cache()
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.exception(f"处理 {entity_name} 时出错")
            return {"error": str(e)}
        finally:
            event.set()
            if entity_name in self.processing_requests:
                del self.processing_requests[entity_name]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        node_status = []
        for node in self.nodes:
            node_status.append({
                "url": node.url,
                "healthy": node.is_available(),
                "consecutive_failures": node.consecutive_failures
            })
        
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "requests_since_last_save": self.requests_since_last_save,
            "seconds_since_last_save": int(time.time() - self.last_save_time),
            "nodes": node_status
        }

    async def force_save_cache(self):
        """强制保存缓存"""
        await self._save_cache_to_file()

    async def close(self):
        """关闭并保存缓存"""
        logger.info("正在关闭搜索管理器，保存缓存...")
        # 关闭前保存缓存
        self._save_cache_to_file_sync()
        self.process_executor.shutdown()
        if self.session:
            await self.session.close()


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(title="Distributed Wikipedia Search Master (Firefox Sandbox)")
search_manager = DistributedSearchManager(AGENT_NODES, MAX_QPM, CACHE_FILE_PATH)


@app.on_event("shutdown")
async def shutdown_event():
    await search_manager.close()


class SearchResponse(BaseModel):
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    stats: Dict[str, Any]


@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(..., description="Entity name")):
    """搜索 Wikipedia 实体"""
    logger.info(f"收到搜索请求: {query}")
    result = await search_manager.get_result(query)
    return SearchResponse(
        output=result.get("output"),
        error=result.get("error"),
        metadata=result.get("metadata")
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取系统统计信息"""
    return StatsResponse(stats=search_manager.get_stats())


@app.get("/health")
async def health_check():
    """健康检查"""
    stats = search_manager.get_stats()
    healthy_nodes = sum(1 for n in stats["nodes"] if n["healthy"])
    return {
        "status": "healthy" if healthy_nodes > 0 else "degraded",
        "healthy_nodes": healthy_nodes,
        "total_nodes": len(stats["nodes"]),
        "cache_size": stats["cache_size"]
    }


@app.post("/save_cache")
async def save_cache():
    """手动触发缓存保存"""
    await search_manager.force_save_cache()
    return {"status": "success", "cache_size": len(search_manager.cache)}


@app.get("/cache_info")
async def cache_info():
    """获取缓存信息"""
    return {
        "cache_size": len(search_manager.cache),
        "cache_file_path": CACHE_FILE_PATH,
        "requests_since_last_save": search_manager.requests_since_last_save,
        "seconds_since_last_save": int(time.time() - search_manager.last_save_time),
        "save_interval_requests": CACHE_SAVE_INTERVAL_REQUESTS,
        "save_interval_seconds": CACHE_SAVE_INTERVAL_SECONDS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020, workers=1)