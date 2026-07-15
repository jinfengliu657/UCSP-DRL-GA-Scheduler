"""
数据库连接和操作封装
"""
import os
import pymysql
from typing import List, Dict, Any
from contextlib import contextmanager


class Database:
    """MySQL数据库连接类"""

    def __init__(self, config: Dict[str, str]):
        """
        初始化数据库连接

        Args:
            config: 数据库配置字典，包含host, port, database, user, password
        """
        self.config = config
        self.connection = None

    def connect(self):
        """建立数据库连接"""
        self.connection = pymysql.connect(
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 3306),
            database=self.config.get('database', 'test1'),
            user=self.config.get('user', 'root'),
            password=os.getenv('DB_PASSWORD', self.config.get('password', '')),
            charset='utf8mb4'
        )
        return self.connection

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None

    @contextmanager
    def get_cursor(self):
        """获取数据库游标的上下文管理器"""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        try:
            yield cursor
        finally:
            cursor.close()

    def fetch_all(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple = None) -> Dict[str, Any]:
        """执行查询并返回单条结果"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchone()

    def execute(self, sql: str, params: tuple = None) -> int:
        """执行SQL语句并返回影响的行数"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            self.connection.commit()
            return cursor.rowcount

    def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """批量执行SQL语句"""
        with self.get_cursor() as cursor:
            cursor.executemany(sql, params_list)
            self.connection.commit()
            return cursor.rowcount
