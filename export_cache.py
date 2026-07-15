"""
本地数据脱水脚本 (export_cache.py)
注意：请在【本地电脑】运行此脚本！
"""
"""
本地数据脱水脚本 (export_cache.py)

功能：
1. 使用 DataLoader 按原始实例划分逻辑加载数据
2. 将每个实例保存为 pkl
3. 保存位置：data/cache/
4. 不修改任何实例划分逻辑

注意：请在【本地有数据库的电脑】运行！
"""

import os
import pickle
from data.data_loader import DataLoader


# 请确保这里的数据库配置与本地一致
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', ''),
    'database': 'test3'
}


# 与 run_ablation.py 完全一致的实例列表
INSTANCES = [
    "S1_Small",
    "S2_Small",
    "S3_Small",
    "S4_Small",
    "M1_Medium",
    "M2_Medium",
    "M3_Medium",
    "L1_Large"
]


def main():

    print("⏳ 正在连接本地 MySQL，开始导出全部实例缓存...")

    # 创建缓存目录
    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    for target_instance in INSTANCES:

        print("\n" + "=" * 60)
        print(f"⏳ 正在提取实例数据: {target_instance}")

        # 这一行保持原始逻辑
        loader = DataLoader(target_instance, DB_CONFIG)
        data = loader.load()

        cache_filename = os.path.join(cache_dir, f"{target_instance}_cache.pkl")

        with open(cache_filename, 'wb') as f:
            pickle.dump(data, f)

        size_mb = os.path.getsize(cache_filename) / (1024 * 1024)

        print(f"✅ 成功！实例 {target_instance} 脱水完成")
        print(f"📦 数据已保存: {cache_filename}")
        print(f"📦 文件大小: {size_mb:.2f} MB")

        # 输出规模信息（方便检查）
        print(
            f"📊 规模统计 -> "
            f"Tasks: {data.num_tasks}, "
            f"Teachers: {data.num_teachers}, "
            f"Classrooms: {data.num_classrooms}, "
            f"Classes: {data.num_classes}"
        )

    print("\n" + "=" * 60)
    print("🎉 所有实例缓存导出完成！")
    print("👉 请将 data/cache/ 文件夹上传至服务器！")


if __name__ == "__main__":
    main()








# /////////////////////生成LI实例   运行成功
# import os
# import pickle
# from data.data_loader import DataLoader
#
# # 请通过 DB_PASSWORD 环境变量提供本地数据库密码
# DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
#
#
# def main():
#     target_instance = 'L1_Large'
#     print(f"⏳ 正在连接本地 MySQL，提取 {target_instance} 数据...")
#
#     loader = DataLoader(target_instance, DB_CONFIG)
#     data = loader.load()
#
#     cache_filename = f"{target_instance}_cache.pkl"
#     with open(cache_filename, 'wb') as f:
#         pickle.dump(data, f)
#     cache_dir = os.path.join("data", "cache")
#     os.makedirs(cache_dir, exist_ok=True)
#
#     cache_filename = os.path.join(cache_dir, f"{target_instance}_cache.pkl")
#
#     with open(cache_filename, 'wb') as f:
#         pickle.dump(data, f)
#
#     size_mb = os.path.getsize(cache_filename) / (1024 * 1024)
#     print(f"✅ 成功！脱水完成。数据已封存至: {cache_filename} (大小: {size_mb:.2f} MB)")
#     print("👉 请将此文件连同项目一起上传至服务器！")
#
#
# if __name__ == "__main__":
#     main()
