#2 日期处理工具\date_utils.py
from datetime import datetime, timedelta
import re

def parse_issue(issue_str):
    """解析期号字符串"""
    match = re.match(r"(\d{8})-(\d{4})", issue_str)
    if not match:
        raise ValueError("无效的期号格式")
    return match.group(1), int(match.group(2))

def get_next_issue(current_issue):
    """获取下一期号"""
    date_str, period = parse_issue(current_issue)
    date = datetime.strptime(date_str, "%Y%m%d")
    
    if period == 1440:
        new_date = date + timedelta(days=1)
        new_period = 1
    else:
        new_date = date
        new_period = period + 1
    
    return f"{new_date.strftime('%Y%m%d')}-{new_period:04d}"