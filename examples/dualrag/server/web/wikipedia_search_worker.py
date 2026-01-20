#!/usr/bin/env python3
"""
apt update

apt install firefox

apt install -y libgtk-3-0t64 libdbus-glib-1-2 libx11-xcb1 libxt6t64 libxcomposite1 libasound2t64 libgl1 libpci3

cp geckodriver /usr/bin/
chmod +x /usr/bin/geckodriver
"""

import os
import re
import time
import random
import asyncio
import threading
import atexit
import logging
import urllib.parse
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    ElementClickInterceptedException,
    NoSuchWindowException,
    SessionNotCreatedException,
    InvalidSessionIdException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementNotInteractableException
)
import os
path_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 配置区域 =================
PORT = 8089
POOL_SIZE = 10
HEADLESS = True
PAGE_LOAD_TIMEOUT = 60
SEARCH_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

WIKI_HOME = "https://en.wikipedia.org/wiki/Main_Page"

GECKODRIVER_PATH = f"{path_dir}/geckodriver"
FIREFOX_BINARY_PATH = f"{path_dir}/firefox/firefox"

USE_PROXY = True
# PROXY_HOST = "172.19.135.130"
# PROXY_PORT = 5000
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
# ===========================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WikiAgent")


class DriverStatus(Enum):
    HEALTHY = "healthy"
    NEEDS_REFRESH = "needs_refresh"
    NEEDS_REBUILD = "needs_rebuild"


@dataclass
class FetchResult:
    success: bool
    html: str
    driver_status: DriverStatus
    error: Optional[str] = None


class WikiDriverPool:
    """Wikipedia Firefox Driver 池"""
    
    _instance = None
    _lock = threading.Lock()
    
    FATAL_EXCEPTIONS = (
        InvalidSessionIdException,
        NoSuchWindowException,
        SessionNotCreatedException,
    )
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_config_done'):
            return
        self._config_done = True
        
        self._pool_size = POOL_SIZE
        self._headless = HEADLESS
        self._page_load_timeout = PAGE_LOAD_TIMEOUT
        self._search_timeout = SEARCH_TIMEOUT
        self._max_retries = MAX_RETRIES
        self._retry_delay = RETRY_DELAY
        
        self._driver_pool: Queue = Queue()
        self._initialized = False
        self._init_lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        self._stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'driver_rebuilds': 0,
            'retries': 0
        }
    
    def _get_firefox_options(self) -> FirefoxOptions:
        """获取 Firefox 配置"""
        options = FirefoxOptions()
        
        if self._headless:
            options.add_argument("--headless")
            
        options.add_argument("--width=1920")
        options.add_argument("--height=1080")
        
        options.set_preference("plugin.state.flash", 0)
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        options.set_preference("general.useragent.override", random.choice(user_agents))
        options.set_preference("intl.accept_languages", "en-US,en;q=0.9")
        
        options.set_preference("dom.webdriver.enabled", False)
        options.set_preference('useAutomationExtension', False)
        
        if USE_PROXY:
            options.set_preference("network.proxy.type", 1)
            options.set_preference("network.proxy.http", PROXY_HOST)
            options.set_preference("network.proxy.http_port", PROXY_PORT)
            options.set_preference("network.proxy.ssl", PROXY_HOST)
            options.set_preference("network.proxy.ssl_port", PROXY_PORT)
            options.set_preference(
                "network.proxy.no_proxies_on",
                ",".join([
                    "localhost",
                    "localhost.localdomain",
                    "127.0.0.1",
                    "::1",
                    "0.0.0.0",
                    "[::1]",
                    "*.local",
                    "*.internal",
                    "moz-extension",
                ])
            )
            options.set_preference("network.proxy.allow_hijacking_localhost", False)
        options.binary_location = FIREFOX_BINARY_PATH

        return options

    def _create_driver(self) -> webdriver.Firefox:
        proxy_vars = [
            "http_proxy", "https_proxy",
            "HTTP_PROXY", "HTTPS_PROXY",
            "ALL_PROXY", "all_proxy",
            "NO_PROXY", "no_proxy",
        ]

        for k in proxy_vars:
            os.environ.pop(k, None)

        service = Service(executable_path=GECKODRIVER_PATH)
        options = self._get_firefox_options()

        driver = webdriver.Firefox(service=service, options=options)
        driver.set_page_load_timeout(self._page_load_timeout)
        driver.implicitly_wait(5)

        return driver

    def _check_driver_health(self, driver) -> DriverStatus:
        try:
            _ = driver.current_url
            _ = driver.title
            return DriverStatus.HEALTHY
        except self.FATAL_EXCEPTIONS:
            return DriverStatus.NEEDS_REBUILD
        except WebDriverException:
            return DriverStatus.NEEDS_REFRESH
        except Exception:
            return DriverStatus.NEEDS_REBUILD
    
    def _rebuild_driver(self, old_driver) -> Optional[webdriver.Firefox]:
        with self._rebuild_lock:
            self._stats['driver_rebuilds'] += 1
            try:
                old_driver.quit()
            except Exception:
                pass
            try:
                new_driver = self._create_driver()
                new_driver.get(WIKI_HOME)
                self._human_delay(3, 5)
                logger.info("Driver 重建成功")
                return new_driver
            except Exception as e:
                logger.error(f"Driver 重建失败: {e}")
                return None
    
    def _refresh_driver(self, driver) -> bool:
        try:
            driver.delete_all_cookies()
            driver.get(WIKI_HOME)
            self._human_delay(3, 5)
            return True
        except Exception:
            return False
    
    def _human_delay(self, min_sec: float = 0.5, max_sec: float = 1.5):
        time.sleep(random.uniform(min_sec, max_sec))
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            logger.info(f"正在初始化 {self._pool_size} 个 Firefox 浏览器...")
            
            self._executor = ThreadPoolExecutor(
                max_workers=self._pool_size,
                thread_name_prefix="wiki_worker"
            )
            
            success_count = 0
            for i in range(self._pool_size):
                driver = self._create_driver()
                driver.get(WIKI_HOME)
                self._human_delay(3, 5)
                self._driver_pool.put(driver)
                success_count += 1
                logger.info(f"浏览器 {i+1}/{self._pool_size} 就绪")
                try:
                    ...
                except Exception as e:
                    logger.error(f"浏览器 {i+1} 创建失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            if success_count == 0:
                raise RuntimeError("无法创建任何浏览器实例")
            
            self._initialized = True
            logger.info(f"初始化完成，可用: {self._driver_pool.qsize()}")
    
    @contextmanager
    def _acquire_driver(self, timeout: float = 180):
        driver = None
        try:
            driver = self._driver_pool.get(timeout=timeout)
            status = self._check_driver_health(driver)
            
            if status == DriverStatus.NEEDS_REBUILD:
                new_driver = self._rebuild_driver(driver)
                if new_driver:
                    driver = new_driver
                else:
                    raise RuntimeError("Driver 重建失败")
            elif status == DriverStatus.NEEDS_REFRESH:
                if not self._refresh_driver(driver):
                    new_driver = self._rebuild_driver(driver)
                    if new_driver:
                        driver = new_driver
                    else:
                        raise RuntimeError("Driver 刷新和重建都失败")
            
            yield driver
            
        except Empty:
            raise RuntimeError(f"获取浏览器超时({timeout}s)")
        finally:
            if driver:
                try:
                    status = self._check_driver_health(driver)
                    if status == DriverStatus.NEEDS_REBUILD:
                        new_driver = self._rebuild_driver(driver)
                        if new_driver:
                            self._driver_pool.put(new_driver)
                    else:
                        self._driver_pool.put(driver)
                except Exception:
                    pass
    
    def _go_to_home_and_wait(self, driver) -> bool:
        """导航到主页并等待加载完成"""
        try:
            current_url = driver.current_url
            
            if "Main_Page" not in current_url or "wikipedia.org" not in current_url:
                logger.info("导航到 Wikipedia 主页")
                driver.get(WIKI_HOME)
            
            wait = WebDriverWait(driver, self._search_timeout)
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            
            self._human_delay(2, 3)
            
            if "Wikipedia" not in driver.title:
                logger.warning(f"页面标题不正确: {driver.title}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"导航到主页失败: {e}")
            return False
    
    def _perform_search_with_js(self, driver, entity: str) -> bool:
        """
        使用 JavaScript 执行搜索操作
        这种方式可以避免 StaleElementReferenceException
        """
        try:
            # 等待页面加载完成
            wait = WebDriverWait(driver, self._search_timeout)
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            
            self._human_delay(1, 2)
            
            # 使用 JavaScript 检查搜索框是否存在并执行搜索
            # 这个脚本会：1. 找到搜索框 2. 设置值 3. 提交表单
            js_search_script = """
            function performSearch(query) {
                // 尝试多种方式找到搜索框
                var searchInput = document.getElementById('searchInput');
                if (!searchInput) {
                    searchInput = document.querySelector('input.cdx-text-input__input');
                }
                if (!searchInput) {
                    searchInput = document.querySelector('input.mw-searchInput');
                }
                if (!searchInput) {
                    searchInput = document.querySelector('input[name="search"]');
                }
                if (!searchInput) {
                    searchInput = document.querySelector('input[placeholder="Search Wikipedia"]');
                }
                
                if (!searchInput) {
                    return {success: false, error: 'Search input not found'};
                }
                
                // 清空并设置值
                searchInput.value = '';
                searchInput.value = query;
                
                // 触发 input 事件（某些网站需要）
                searchInput.dispatchEvent(new Event('input', { bubbles: true }));
                searchInput.dispatchEvent(new Event('change', { bubbles: true }));
                
                return {success: true, error: null};
            }
            return performSearch(arguments[0]);
            """
            
            # 执行 JS 设置搜索值
            result = driver.execute_script(js_search_script, entity)
            
            if not result.get('success'):
                logger.error(f"JS 搜索失败: {result.get('error')}")
                return False
            
            logger.info(f"JS 设置搜索值成功: {entity}")
            
            # 模拟人类延迟
            self._human_delay(0.5, 1.0)
            
            # 使用 JS 提交搜索表单
            js_submit_script = """
            function submitSearch() {
                // 方法1: 找到搜索框并按回车
                var searchInput = document.getElementById('searchInput');
                if (!searchInput) {
                    searchInput = document.querySelector('input.cdx-text-input__input');
                }
                if (!searchInput) {
                    searchInput = document.querySelector('input[name="search"]');
                }
                
                if (searchInput) {
                    // 创建并触发 keydown 事件 (Enter 键)
                    var enterEvent = new KeyboardEvent('keydown', {
                        key: 'Enter',
                        code: 'Enter',
                        keyCode: 13,
                        which: 13,
                        bubbles: true
                    });
                    searchInput.dispatchEvent(enterEvent);
                    
                    // 也尝试提交表单
                    var form = searchInput.closest('form');
                    if (form) {
                        form.submit();
                        return {success: true, method: 'form_submit'};
                    }
                    return {success: true, method: 'keydown'};
                }
                
                // 方法2: 直接提交搜索表单
                var searchForm = document.getElementById('searchform');
                if (searchForm) {
                    searchForm.submit();
                    return {success: true, method: 'searchform'};
                }
                
                return {success: false, error: 'Cannot submit search'};
            }
            return submitSearch();
            """
            
            submit_result = driver.execute_script(js_submit_script)
            logger.info(f"JS 提交搜索结果: {submit_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"JS 搜索执行失败: {e}")
            return False
    
    def _perform_search_direct_url(self, driver, entity: str) -> bool:
        """
        备选方案：直接通过 URL 搜索
        如果 JS 方式失败，使用这个方法
        """
        try:
            encoded_entity = urllib.parse.quote(entity)
            # 使用 Wikipedia 的搜索 URL 格式
            search_url = f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_entity}&go=Go"
            
            logger.info(f"使用直接 URL 搜索: {entity}")
            driver.get(search_url)
            
            self._human_delay(2, 4)
            
            return True
            
        except Exception as e:
            logger.error(f"直接 URL 搜索失败: {e}")
            return False
    
    def _wait_for_result(self, driver) -> bool:
        """等待搜索结果"""
        wait = WebDriverWait(driver, self._search_timeout)
        
        try:
            # 等待 URL 变化（说明搜索已提交）
            wait.until(lambda d: "Main_Page" not in d.current_url)
            
            # 等待内容出现
            wait.until(lambda d: 
                d.find_elements(By.ID, "firstHeading") or
                d.find_elements(By.ID, "mw-content-text") or 
                d.find_elements(By.CLASS_NAME, "mw-search-results") or
                d.find_elements(By.CLASS_NAME, "mw-search-nonefound") or
                d.find_elements(By.CSS_SELECTOR, ".mw-parser-output")
            )
            
            self._human_delay(1, 2)
            
            return True
            
        except TimeoutException:
            logger.warning("等待搜索结果超时，但继续尝试获取页面")
            return True
    
    def _check_html_quality(self, html: str) -> Optional[str]:
        """检查 HTML 质量"""
        if not html or len(html) < 1000:
            return "HTML内容过短"
        
        html_lower = html.lower()
        
        # if "unusual traffic" in html_lower:
        #     return "检测到异常流量"
        
        # if "captcha" in html_lower and "recaptcha" in html_lower:
        #     return "触发验证码"
        
        # if "wikipedia" not in html_lower:
        #     return "不是 Wikipedia 页面"
        
        return None
    
    def _fetch_once(self, driver, entity: str) -> FetchResult:
        """单次获取"""
        try:
            logger.info(f"开始搜索: {entity}")
            
            # 1. 确保在主页
            if not self._go_to_home_and_wait(driver):
                return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, "无法加载主页")
            
            # 2. 执行搜索 - 优先使用 JS 方式
            search_success = False
            
            # 尝试 JS 搜索
            if self._perform_search_with_js(driver, entity):
                self._human_delay(2, 4)
                
                # 检查是否已经离开主页
                if "Main_Page" not in driver.current_url:
                    search_success = True
                    logger.info("JS 搜索成功")
            
            # 如果 JS 搜索失败，使用直接 URL
            if not search_success:
                logger.info("JS 搜索未跳转，尝试直接 URL 方式")
                if self._perform_search_direct_url(driver, entity):
                    search_success = True
            
            if not search_success:
                return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, "搜索操作失败")
            
            # 3. 等待结果
            self._wait_for_result(driver)
            
            # 4. 获取 HTML
            try:
                html = driver.page_source
            except Exception as e:
                return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, f"获取HTML失败: {e}")
            
            # 5. 检查质量
            issue = self._check_html_quality(html)
            if issue:
                logger.warning(f"HTML 质量问题: {issue}")
                return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, issue)
            
            logger.info(f"成功获取: {entity} (HTML长度: {len(html)})")
            
            # 6. 回到主页
            try:
                driver.get(WIKI_HOME)
                self._human_delay(1, 2)
            except Exception:
                pass
            
            return FetchResult(True, html, DriverStatus.HEALTHY)
            
        except self.FATAL_EXCEPTIONS as e:
            return FetchResult(False, "", DriverStatus.NEEDS_REBUILD, f"致命错误: {type(e).__name__}")
        except TimeoutException:
            return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, "操作超时")
        except Exception as e:
            logger.exception(f"获取 {entity} 时发生异常")
            return FetchResult(False, "", DriverStatus.NEEDS_REFRESH, str(e))
    
    def _do_fetch(self, entity: str) -> Dict[str, Any]:
        """执行获取（带重试）"""
        self._ensure_initialized()
        self._stats['total_fetches'] += 1
        last_error = None
        
        for attempt in range(self._max_retries):
            try:
                with self._acquire_driver() as driver:
                    result = self._fetch_once(driver, entity)
                    
                    if result.success:
                        self._stats['successful_fetches'] += 1
                        return {"success": True, "html": result.html}
                    
                    last_error = result.error
                    
                    if result.driver_status == DriverStatus.NEEDS_REFRESH:
                        self._refresh_driver(driver)
                    
                    if attempt < self._max_retries - 1:
                        self._stats['retries'] += 1
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.warning(f"获取 '{entity}' 失败({last_error})，{wait_time}秒后重试")
                        time.sleep(wait_time)
                        
            except Exception as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    self._stats['retries'] += 1
                    wait_time = self._retry_delay * (attempt + 1)
                    logger.warning(f"获取 '{entity}' 异常({last_error})，{wait_time}秒后重试")
                    time.sleep(wait_time)
        
        self._stats['failed_fetches'] += 1
        logger.error(f"获取 '{entity}' 最终失败: {last_error}")
        return {"success": False, "error": last_error}
    
    async def fetch_async(self, entity: str) -> Dict[str, Any]:
        """异步获取"""
        loop = asyncio.get_event_loop()
        
        if not self._initialized:
            logger.info("正在后台线程中初始化浏览器池...")
            await loop.run_in_executor(None, self._ensure_initialized)
        
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._pool_size)
        
        async with self._semaphore:
            return await loop.run_in_executor(self._executor, self._do_fetch, entity)
    
    def get_stats(self) -> Dict:
        return {
            **self._stats,
            'pool_size': self._pool_size,
            'available_drivers': self._driver_pool.qsize() if self._initialized else 0,
            'initialized': self._initialized
        }
    
    def shutdown(self):
        if not self._initialized:
            return
        logger.info("正在关闭浏览器池...")
        while not self._driver_pool.empty():
            try:
                driver = self._driver_pool.get_nowait()
                driver.quit()
            except Exception:
                pass
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._initialized = False
        logger.info("浏览器池已关闭")


# ============== 全局实例 ==============

_pool = WikiDriverPool()
atexit.register(_pool.shutdown)


# ============== FastAPI ==============

app = FastAPI(title="Wikipedia Search Agent")


class HtmlResponse(BaseModel):
    html: str
    node_id: str
    error: Optional[str] = None


class StatsResponse(BaseModel):
    stats: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    logger.info("Agent 启动，开始预热浏览器池...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _pool._ensure_initialized)
    logger.info("浏览器池预热完成")


@app.on_event("shutdown")
async def shutdown_event():
    _pool.shutdown()


@app.get("/fetch", response_model=HtmlResponse)
async def fetch_wiki(entity: str = Query(..., description="Entity name")):
    logger.info(f"收到请求: {entity}")
    
    try:
        result = await _pool.fetch_async(entity)
        
        if result.get("success"):
            return HtmlResponse(
                html=result["html"],
                node_id=os.uname().nodename
            )
        else:
            raise HTTPException(
                status_code=502, 
                detail=f"Fetch failed: {result.get('error', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取 {entity} 时发生错误")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    return StatsResponse(stats=_pool.get_stats())


@app.get("/health")
async def health_check():
    stats = _pool.get_stats()
    return {
        "status": "healthy" if stats["initialized"] else "initializing",
        "available_drivers": stats["available_drivers"],
        "pool_size": stats["pool_size"],
        "stats": stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=True)