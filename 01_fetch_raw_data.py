"""
FinAgent 数据拉取脚本 - Day 5
用途：批量拉取沪深300研报 + 财务数据，构建检索语料库
环境：Google Colab
依赖：pip install akshare pandas tqdm

运行方式：
    python 01_fetch_raw_data.py                    # 拉取全部沪深300
    python 01_fetch_raw_data.py --max_stocks 10    # 先拉10只测试
    python 01_fetch_raw_data.py --skip_reports     # 只拉财务数据
    python 01_fetch_raw_data.py --skip_financial   # 只拉研报
"""

import akshare as ak
import pandas as pd
import json
import time
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
OUTPUT_DIR = "./data/raw"
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
FINANCIAL_DIR = os.path.join(OUTPUT_DIR, "financial")
LOG_DIR = "./logs"

SLEEP_BETWEEN_STOCKS = 1.0      # 每只股票间隔（秒）
SLEEP_BETWEEN_REQUESTS = 0.5    # 每个API请求间隔（秒）
MAX_RETRIES = 3                 # 单个请求最大重试次数

# 创建目录
for d in [REPORT_DIR, FINANCIAL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"fetch_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ Step 1: 获取沪深300成分股列表 ============

def get_hs300_stocks() -> pd.DataFrame:
    """
    获取沪深300成分股列表
    尝试多个接口，确保至少一个能用
    """
    # 方案1: 中证指数官网
    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        logger.info(f"[csindex] 获取沪深300成分股 {len(df)} 只")
        # 统一字段名
        if '成分券代码' in df.columns:
            df = df.rename(columns={'成分券代码': 'stock_code', '成分券名称': 'stock_name'})
        elif '证券代码' in df.columns:
            df = df.rename(columns={'证券代码': 'stock_code', '证券简称': 'stock_name'})
        return df[['stock_code', 'stock_name']].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[csindex] 失败: {e}")

    # 方案2: 东财接口
    try:
        df = ak.index_stock_cons(symbol="000300")
        logger.info(f"[eastmoney] 获取沪深300成分股 {len(df)} 只")
        if '品种代码' in df.columns:
            df = df.rename(columns={'品种代码': 'stock_code', '品种名称': 'stock_name'})
        return df[['stock_code', 'stock_name']].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[eastmoney] 失败: {e}")

    # 方案3: 从本地文件读取（备用）
    fallback_path = os.path.join(OUTPUT_DIR, "hs300_stocks.csv")
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path, dtype={'stock_code': str})
        logger.info(f"[本地文件] 读取沪深300成分股 {len(df)} 只")
        return df

    raise RuntimeError("无法获取沪深300成分股列表，请手动准备 hs300_stocks.csv")


# ============ Step 2: 拉取研报数据 ============

def fetch_reports_for_stock(stock_code: str) -> pd.DataFrame:
    """
    拉取单只股票的研报数据
    返回: DataFrame，包含研报标题、评级、机构、盈利预测等
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_research_report_em(symbol=stock_code)
            if df is not None and len(df) > 0:
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 2
                logger.warning(f"  [{stock_code}] 研报请求失败(第{attempt+1}次): {e}, {wait}秒后重试")
                time.sleep(wait)
            else:
                logger.error(f"  [{stock_code}] 研报请求最终失败: {e}")
                return pd.DataFrame()


def batch_fetch_reports(stocks: pd.DataFrame) -> dict:
    """
    批量拉取研报数据
    返回: {stock_code: DataFrame} 的字典
    """
    results = {}
    stats = {"success": 0, "empty": 0, "failed": 0, "total_reports": 0}

    logger.info(f"===== 开始拉取研报数据，共 {len(stocks)} 只股票 =====")

    for idx, row in tqdm(stocks.iterrows(), total=len(stocks), desc="拉取研报"):
        code = row['stock_code']
        name = row['stock_name']

        df = fetch_reports_for_stock(code)

        if df is not None and len(df) > 0:
            # 添加股票名称（原始数据可能没有）
            if '股票简称' not in df.columns:
                df['股票简称'] = name
            results[code] = df
            stats["success"] += 1
            stats["total_reports"] += len(df)
            logger.info(f"  [{code} {name}] 获取 {len(df)} 篇研报")
        else:
            stats["empty"] += 1
            logger.info(f"  [{code} {name}] 无研报数据")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # 每50只股票保存一次checkpoint
        if (idx + 1) % 50 == 0:
            _save_report_checkpoint(results, stats)

    # 最终保存
    _save_report_checkpoint(results, stats)
    logger.info(f"===== 研报拉取完成: 成功{stats['success']}, 空{stats['empty']}, "
                f"失败{stats['failed']}, 共{stats['total_reports']}篇 =====")
    return results


def _save_report_checkpoint(results: dict, stats: dict):
    """保存研报数据checkpoint"""
    if not results:
        return
    all_reports = pd.concat(results.values(), ignore_index=True)
    save_path = os.path.join(REPORT_DIR, "all_reports.csv")
    all_reports.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    # 保存统计信息
    with open(os.path.join(REPORT_DIR, "fetch_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  [checkpoint] 已保存 {len(all_reports)} 篇研报到 {save_path}")


# ============ Step 3: 拉取财务数据 ============

def fetch_financial_for_stock(stock_code: str) -> dict:
    """
    拉取单只股票的关键财务数据
    
    改用 stock_financial_analysis_indicator（东财源）统一拉取。
    原方案用4个接口（利润表/资产负债表/现金流/同花顺指标），但：
    - stock_profit_sheet_by_report_em 已失效（东财网页结构变更）
    - stock_financial_abstract_ths 是同花顺源，反爬严格且数据格式不统一
    
    stock_financial_analysis_indicator 一个接口包含86个字段，覆盖：
    盈利能力（ROE/毛利率/净利率）、成长能力（营收增长率/利润增长率）、
    偿债能力（资产负债率/流动比率）、运营效率（周转率）、每股指标（EPS/BPS）、
    总资产等。完全满足需求，且只需一次API调用，更稳定。
    
    面试追问：为什么不直接用结构化数据做检索？
    答：向量检索对结构化数据不友好。
    "ROE: 15.3%"转成"贵州茅台2024年ROE为15.3%"后，
    query="茅台盈利能力"就能检索到。这是一个工程上的trade-off。
    """
    result = {}

    try:
        df = ak.stock_financial_analysis_indicator(
            symbol=stock_code, start_year="2022"
        )
        if df is not None and len(df) > 0:
            # 只保留年报(12-31)和半年报(06-30)，去掉Q1(03-31)和Q3(09-30)
            # 原因：Q1/Q3的指标是单季或前N季累计，容易误导
            # 例如Q1的ROE只有一个季度，用户会误以为是全年
            # 年报是全年累计数据最准确，半年报做同比分析有用
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[df['日期'].dt.month.isin([6, 12])].copy()
            
            if len(df) > 0:
                records = df.head(6).to_dict(orient='records')
                # 日期转回字符串便于JSON序列化
                for r in records:
                    r['日期'] = r['日期'].strftime('%Y-%m-%d')
                result["financial_indicators"] = records
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    except Exception as e:
        result["financial_error"] = str(e)

    return result


def batch_fetch_financial(stocks: pd.DataFrame) -> dict:
    """
    批量拉取财务数据
    返回: {stock_code: dict} 的字典
    """
    results = {}
    stats = {"success": 0, "partial": 0, "failed": 0}

    logger.info(f"===== 开始拉取财务数据，共 {len(stocks)} 只股票 =====")

    for idx, row in tqdm(stocks.iterrows(), total=len(stocks), desc="拉取财务"):
        code = row['stock_code']
        name = row['stock_name']

        data = fetch_financial_for_stock(code)

        # 统计成功/失败
        has_data = "financial_indicators" in data
        has_error = "financial_error" in data

        if has_data and not has_error:
            stats["success"] += 1
        elif has_data:
            stats["partial"] += 1
        else:
            stats["failed"] += 1

        data["stock_code"] = code
        data["stock_name"] = name
        results[code] = data

        n_periods = len(data.get("financial_indicators", []))
        logger.info(f"  [{code} {name}] 财务指标: "
                    f"{'✓ ' + str(n_periods) + '期' if has_data else '✗'}"
                    f"{' (有错误)' if has_error else ''}")

        time.sleep(SLEEP_BETWEEN_STOCKS)

        # 每50只股票保存一次checkpoint
        if (idx + 1) % 50 == 0:
            _save_financial_checkpoint(results, stats)

    # 最终保存
    _save_financial_checkpoint(results, stats)
    logger.info(f"===== 财务数据拉取完成: 成功{stats['success']}, "
                f"部分{stats['partial']}, 失败{stats['failed']} =====")
    return results


def _save_financial_checkpoint(results: dict, stats: dict):
    """保存财务数据checkpoint"""
    if not results:
        return
    save_path = os.path.join(FINANCIAL_DIR, "all_financial.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(FINANCIAL_DIR, "fetch_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"  [checkpoint] 已保存 {len(results)} 只股票财务数据到 {save_path}")


# ============ Step 4: 断点续传支持 ============

def load_existing_progress() -> set:
    """
    检查已有数据，返回已完成的股票代码集合
    避免重复拉取（Colab可能中途断开）
    """
    done_codes = set()

    # 检查研报
    report_path = os.path.join(REPORT_DIR, "all_reports.csv")
    if os.path.exists(report_path):
        df = pd.read_csv(report_path, dtype={'股票代码': str})
        if '股票代码' in df.columns:
            done_codes.update(df['股票代码'].unique())
            logger.info(f"[断点续传] 已有 {len(done_codes)} 只股票的研报数据")

    return done_codes


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent 数据拉取")
    parser.add_argument("--max_stocks", type=int, default=0, help="最多拉取股票数（0=全部）")
    parser.add_argument("--skip_reports", action="store_true", help="跳过研报拉取")
    parser.add_argument("--skip_financial", action="store_true", help="跳过财务数据拉取")
    parser.add_argument("--resume", action="store_true", help="断点续传模式")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FinAgent 数据拉取开始")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # 1. 获取股票列表
    stocks = get_hs300_stocks()
    
    # 保存股票列表（备用）
    stocks.to_csv(os.path.join(OUTPUT_DIR, "hs300_stocks.csv"), index=False, encoding='utf-8-sig')
    logger.info(f"沪深300成分股: {len(stocks)} 只")

    # 2. 断点续传：过滤已完成的
    if args.resume:
        done = load_existing_progress()
        stocks = stocks[~stocks['stock_code'].isin(done)].reset_index(drop=True)
        logger.info(f"断点续传: 跳过 {len(done)} 只, 剩余 {len(stocks)} 只")

    # 3. 限制数量（测试用）
    if args.max_stocks > 0:
        stocks = stocks.head(args.max_stocks)
        logger.info(f"测试模式: 只拉取前 {args.max_stocks} 只")

    # 4. 拉取研报
    if not args.skip_reports:
        report_results = batch_fetch_reports(stocks)

    # 5. 拉取财务数据
    if not args.skip_financial:
        financial_results = batch_fetch_financial(stocks)

    # 6. 汇总统计
    logger.info("=" * 60)
    logger.info("数据拉取完成！")
    logger.info(f"数据保存目录: {os.path.abspath(OUTPUT_DIR)}")
    logger.info(f"  研报: {REPORT_DIR}/all_reports.csv")
    logger.info(f"  财务: {FINANCIAL_DIR}/all_financial.json")
    logger.info("=" * 60)
    logger.info("下一步: 运行 02_download_pdfs.py 下载研报PDF")


if __name__ == "__main__":
    main()
