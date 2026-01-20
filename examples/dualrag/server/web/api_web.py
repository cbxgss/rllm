from typing import Any
import aiohttp
from tenacity import retry, stop_after_attempt, stop_never, wait_random_exponential


MASTER_URL = "http://127.0.0.1:8020/search"


@retry(reraise=True, stop=stop_never, wait=wait_random_exponential(min=1, max=10))
async def search_entity(entity: str, timeout: int = 180) -> dict[str, Any]:
    """
    异步查询 Wikipedia 实体
    :param entity: 实体名，例如 "Python_(programming_language)"
    :param timeout: 请求超时时间（秒）
    :return: 返回 Master Server 的 JSON 响应
    """
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        async with session.get(MASTER_URL, params={"query": entity}) as resp:
            data = await resp.json()
            return data


if __name__ == "__main__":
    import asyncio

    async def main():
        entity = "Python_(programming_language)"
        res = await search_entity(entity)
        print(res["output"])
        print(len(res["output"]))

    asyncio.run(main())
