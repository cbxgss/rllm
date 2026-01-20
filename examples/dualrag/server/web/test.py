#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import statistics
import random

MASTER_URL = "http://127.0.0.1:8020/search"

CONCURRENCY = 10
TIMEOUT = 180

SAME_QUERY = False

QUERY_POOL = [
    "Python_(programming_language)",
    "Java_(programming_language)",
    "C++",
    "JavaScript",
    "Linux",
    "Elon_Musk",
    "Joe_Biden",
    "Donald_Trump",
    "Kamala_Harris",
    "Barack_Obama",
    "Taylor_Swift",
    "Cristiano_Ronaldo",
    "Lionel_Messi",
    "World_War_II",
    "United_States",
    "China",
    "India",
    "Artificial_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Neural_network",
    "Oppenheimer_(film)",
    "Dune:_Part_Two",
    "The_Batman_(film)",
    "2024_Summer_Olympics",
    "UEFA_Euro_2024",
    "2024_United_States_presidential_election",
    "2024_Indian_general_election",
    "Project_2025",
    "ChatGPT",
    "Linux_kernel",
    "OpenAI",
    "Tesla,_Inc.",
    "SpaceX",
    "Jeff_Bezos",
    "Bill_Gates",
    "Mark_Zuckerberg",
    "Facebook",
    "Twitter",
    "Instagram",
    "Reddit",
    "Stack_Overflow",
    "Wikipedia",
    "Python_(mythology)",
    "Leonardo_da_Vinci",
    "Albert_Einstein",
    "Isaac_Newton",
    "Marie_Curie",
    "Stephen_Hawking",
    "Nikola_Tesla",
    "Ada_Lovelace",
    "Alan_Turing",
    "Grace_Hopper",
    "Sundar_Pichai",
    "Satya_Nadella",
    "Larry_Page",
    "Sergey_Brin",
    "Bill_Clinton",
    "Franklin_D._Roosevelt",
    "Abraham_Lincoln",
    "George_Washington",
    "Thomas_Jefferson",
    "World_War_I",
    "Cold_War",
    "Industrial_Revolution",
    "Renaissance",
    "French_Revolution",
    "American_Civil_War",
    "Great_Depression",
    "COVID-19_pandemic",
    "2019–20_coronavirus_pandemic",
    "2020_United_States_presidential_election",
    "2020_Summer_Olympics",
    "Premier_League",
    "La_Liga",
    "Bundesliga",
    "Serie_A_(football)",
    "UEFA_Champions_League",
    "FIFA_World_Cup",
    "NBA",
    "NFL",
    "MLB",
    "NHL",
    "Formula_One",
    "Tour_de_France",
    "Wimbledon",
    "US_Open_(tennis)",
    "Oscar_award",
    "Golden_Globe_Award",
    "Grammy_Award",
    "Emmy_Award",
    "Game_of_Thrones",
    "Breaking_Bad",
    "Stranger_Things",
    "The_Lord_of_the_Rings",
    "Harry_Potter",
    "Star_Wars",
    "Marvel_Cinematic_Universe",
    "DC_Extended_Universe",
    "Black_Panther_(film)",
    "Avengers:_Endgame",
    "Spider-Man:_No_Way_Home",
    "Doctor_Strange_in_the_Multiverse_of_Madness",
    "The_Batman_(2022_film)",
    "Avatar:_The_Way_of_Water"
]

async def worker(idx, session, start_event, stats):
    query = random.choice(QUERY_POOL)

    await start_event.wait()  # 🔫 等待统一起跑

    start = time.time()
    try:
        async with session.get(
            MASTER_URL,
            params={"query": query},
        ) as resp:
            data = await resp.json()
            latency = time.time() - start

            if resp.status == 200 and not data.get("error"):
                stats["success"] += 1
            else:
                stats["fail"] += 1

            stats["latencies"].append(latency)

    except Exception:
        latency = time.time() - start
        stats["fail"] += 1
        stats["latencies"].append(latency)


async def main():
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    # 🔥 关键：放开连接池
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY)

    stats = {
        "success": 0,
        "fail": 0,
        "latencies": []
    }

    start_event = asyncio.Event()

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector
    ) as session:

        tasks = [
            asyncio.create_task(worker(i, session, start_event, stats))
            for i in range(CONCURRENCY)
        ]

        # 确保所有 task 都 ready
        await asyncio.sleep(0.1)

        start_time = time.time()
        start_event.set()   # 🚀 所有请求同时出发
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    # 结果统计
    lat = stats["latencies"]

    print("\n========== 100 并发压力测试 ==========")
    print(f"并发请求数: {CONCURRENCY}")
    print(f"成功请求  : {stats['success']}")
    print(f"失败请求  : {stats['fail']}")
    print(f"总耗时    : {total_time:.2f}s")
    print(f"瞬时 QPS  : {CONCURRENCY / total_time:.2f}")

    if lat:
        print("\n--- 延迟统计 (秒) ---")
        print(f"平均延迟 : {statistics.mean(lat):.2f}")
        print(f"P95 延迟 : {statistics.quantiles(lat, n=20)[18]:.2f}")
        print(f"最大延迟 : {max(lat):.2f}")
    print("=====================================\n")


if __name__ == "__main__":
    asyncio.run(main())
