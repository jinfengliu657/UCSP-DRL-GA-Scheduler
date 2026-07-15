"""
导入SQL数据到MySQL
"""
import os
import pymysql
import re
from pathlib import Path

sql_file = Path(__file__).resolve().with_name("test3.sql")

# 读取SQL文件
with open(sql_file, 'r', encoding='utf-8') as f:
    sql_content = f.read()

# 连接MySQL
conn = pymysql.connect(
    host="localhost",
    user="root",
    password=os.getenv("DB_PASSWORD", ""),
    charset="utf8mb4"
)
cursor = conn.cursor()

try:
    # 创建数据库（如果不存在）
    print("创建数据库 test3...")
    cursor.execute("CREATE DATABASE IF NOT EXISTS test3 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute("USE test3")
    print("数据库创建成功！")

    # 执行SQL语句
    print("开始导入数据...")
    # 分割SQL语句（以分号分隔）
    statements = []
    for statement in sql_content.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            statements.append(statement)

    for i, statement in enumerate(statements):
        if statement:
            try:
                cursor.execute(statement)
            except Exception as e:
                # 忽略已存在的表等错误
                if "Duplicate" not in str(e) and "already exists" not in str(e):
                    print(f"警告: {e}")

    conn.commit()
    print(f"导入完成！共执行 {len(statements)} 条语句")

except Exception as e:
    print(f"导入失败: {e}")
finally:
    cursor.close()
    conn.close()
