"""
FinAgent Chunk 构建脚本 - Day 5
用途：将研报 + 财务原始数据转化为检索友好的 chunks
环境：Google Colab
依赖：pip install pandas tqdm

运行方式：
    python 04_build_chunks.py                         # 处理全部数据
    python 04_build_chunks.py --report_only           # 只处理研报
    python 04_build_chunks.py --financial_only        # 只处理财务

面试追问：你的chunk是怎么设计的？
答：研报数据本身是摘要级别（标题+评级+盈利预测），天然适合做单条chunk。
财务数据是结构化的，我转成自然语言描述后作为chunk，这样向量检索能匹配到。
比如query="茅台盈利能力"能匹配到"贵州茅台2024年ROE为15.3%"。
"""

import pandas as pd
import json
import os
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
RAW_DIR = "./data/raw"
CHUNK_DIR = "./data/processed"
os.makedirs(CHUNK_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ 研报 Chunk 构建 ============

def build_report_chunks(report_path: str) -> list[dict]:
    """
    将研报数据转为检索chunks
    
    每篇研报 → 1个chunk，包含：
    - 报告标题（核心语义信息）
    - 机构 + 评级（机构观点）
    - 盈利预测数据（硬数据）
    - 行业 + 日期（元信息）
    
    面试追问：为什么不每个字段单独做chunk？
    答：研报摘要本身就很短（一行标题+几个字段），拆开后每个chunk语义太稀疏，
    检索时会匹配到大量无关结果。合在一起语义更完整。
    """
    df = pd.read_csv(report_path, dtype={'股票代码': str})
    chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建研报chunks"):
        code = str(row.get('股票代码', '')).strip()
        name = str(row.get('股票简称', '')).strip()
        title = str(row.get('报告名称', '')).strip()
        rating = str(row.get('东财评级', '')).strip()
        institution = str(row.get('机构', '')).strip()
        industry = str(row.get('行业', '')).strip()
        date = str(row.get('日期', '')).strip()

        # 构建自然语言描述
        text_parts = [f"{name}({code}) 研报：{title}"]
        
        if institution and institution != 'nan':
            text_parts.append(f"出品机构：{institution}")
        if rating and rating != 'nan':
            text_parts.append(f"评级：{rating}")

        # 盈利预测
        eps_parts = []
        for year in ['2025', '2026', '2027']:
            eps = row.get(f'{year}-盈利预测-收益')
            pe = row.get(f'{year}-盈利预测-市盈率')
            if pd.notna(eps) and pd.notna(pe):
                eps_parts.append(f"{year}年预测EPS {eps}元，PE {pe}倍")
        if eps_parts:
            text_parts.append("盈利预测：" + "；".join(eps_parts))

        text = "\n".join(text_parts)

        chunk = {
            "text": text,
            "metadata": {
                "source_type": "report",
                "stock_code": code,
                "stock_name": name,
                "institution": institution,
                "rating": rating,
                "industry": industry,
                "date": date,
                "report_title": title,
            }
        }
        chunks.append(chunk)

    logger.info(f"研报chunks: {len(chunks)} 条")
    return chunks


# ============ 财务数据 Chunk 构建 ============

def build_financial_chunks(financial_path: str) -> list[dict]:
    """
    将 stock_financial_analysis_indicator 的结构化数据转为自然语言chunks
    
    数据来源：ak.stock_financial_analysis_indicator，每期包含86个字段
    
    策略：每期数据生成2个chunk（盈利+结构），而不是原来的4个chunk（利润表/资产负债/现金流/指标）。
    原因：新接口的数据已经是汇总指标，不需要按报表拆分。合并后每个chunk语义更完整，
    检索时一次就能拿到盈利能力+成长性的全貌。
    
    面试追问：为什么转成自然语言而不是直接存结构化数据？
    答：因为Agent通过自然语言检索，向量检索对结构化数据不友好。
    将"ROE: 15.3%"转成"贵州茅台2024年ROE为15.3%"后，
    query="茅台盈利能力"就能检索到。这是一个工程上的trade-off。
    """
    with open(financial_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    chunks = []

    for code, data in tqdm(all_data.items(), desc="构建财务chunks"):
        name = data.get("stock_name", code)

        if "financial_indicators" not in data:
            continue

        for record in data["financial_indicators"]:
            date = str(record.get("日期", "未知日期"))
            
            # 标注报告期类型，避免Agent混淆年报和半年报
            if date.endswith("12-31"):
                period_label = "年报"
            elif date.endswith("06-30"):
                period_label = "半年报"
            else:
                period_label = ""

            # Chunk 1: 盈利能力 + 成长性 + 每股指标
            text = _profitability_to_text(code, name, date, record, period_label)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source_type": "financial",
                        "data_type": "profitability",
                        "stock_code": code,
                        "stock_name": name,
                        "report_date": date,
                    }
                })

            # Chunk 2: 偿债能力 + 运营效率 + 资产结构
            text = _structure_to_text(code, name, date, record, period_label)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source_type": "financial",
                        "data_type": "balance_structure",
                        "stock_code": code,
                        "stock_name": name,
                        "report_date": date,
                    }
                })

    logger.info(f"财务chunks: {len(chunks)} 条")
    return chunks


def _safe_fmt(value, suffix="", decimals=2) -> str:
    """安全格式化数值，处理NaN和None"""
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v):
            return None
        return f"{v:.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return None


def _format_amount(value, unit="元") -> str:
    """将大数字转为可读格式：亿/万"""
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        return None
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f}亿{unit}"
    elif abs(v) >= 1e4:
        return f"{v/1e4:.2f}万{unit}"
    else:
        return f"{v:.2f}{unit}"


def _profitability_to_text(code: str, name: str, date: str, record: dict, period_label: str = "") -> str:
    """
    盈利能力 + 成长性 + 每股指标 → 自然语言
    
    对应Agent查询场景：
    - "XX公司盈利能力怎么样" → 匹配 ROE/毛利率/净利率
    - "XX公司业绩增长情况" → 匹配 营收增长率/净利润增长率
    - "XX公司每股收益" → 匹配 EPS/BPS
    """
    header = f"{name}({code}) {date}"
    if period_label:
        header += f"({period_label})"
    parts = [f"{header}盈利与成长指标"]
    valid = False

    # 盈利能力
    fields = [
        ("净资产收益率(%)", "净资产收益率(ROE)", "%"),
        ("加权净资产收益率(%)", "加权ROE", "%"),
        ("主营业务利润率(%)", "主营业务利润率(毛利率)", "%"),
        ("销售净利率(%)", "销售净利率", "%"),
        ("营业利润率(%)", "营业利润率", "%"),
        ("总资产利润率(%)", "总资产利润率(ROA)", "%"),
    ]
    for key, label, suffix in fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    # 成长性
    growth_fields = [
        ("主营业务收入增长率(%)", "营收增长率", "%"),
        ("净利润增长率(%)", "净利润增长率", "%"),
        ("净资产增长率(%)", "净资产增长率", "%"),
        ("总资产增长率(%)", "总资产增长率", "%"),
    ]
    for key, label, suffix in growth_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    # 每股指标
    eps_fields = [
        ("摊薄每股收益(元)", "每股收益(EPS)", "元"),
        ("每股净资产_调整后(元)", "每股净资产(BPS)", "元"),
        ("每股经营性现金流(元)", "每股经营现金流", "元"),
        ("每股未分配利润(元)", "每股未分配利润", "元"),
    ]
    for key, label, suffix in eps_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    # 总资产（方便Agent做规模判断）
    total_assets = _format_amount(record.get("总资产(元)"))
    if total_assets:
        parts.append(f"总资产{total_assets}")
        valid = True

    return "，".join(parts) + "。" if valid else None


def _structure_to_text(code: str, name: str, date: str, record: dict, period_label: str = "") -> str:
    """
    偿债能力 + 运营效率 + 资产结构 → 自然语言
    
    对应Agent查询场景：
    - "XX公司财务风险" → 匹配 资产负债率/流动比率
    - "XX公司运营效率" → 匹配 周转率/周转天数
    - "XX公司资产结构" → 匹配 产权比率/股东权益比率
    """
    header = f"{name}({code}) {date}"
    if period_label:
        header += f"({period_label})"
    parts = [f"{header}偿债与运营指标"]
    valid = False

    # 偿债能力
    fields = [
        ("资产负债率(%)", "资产负债率", "%"),
        ("流动比率", "流动比率", ""),
        ("速动比率", "速动比率", ""),
        ("现金比率(%)", "现金比率", "%"),
        ("股东权益比率(%)", "股东权益比率", "%"),
        ("产权比率(%)", "产权比率", "%"),
    ]
    for key, label, suffix in fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    # 运营效率
    efficiency_fields = [
        ("总资产周转率(次)", "总资产周转率", "次"),
        ("存货周转率(次)", "存货周转率", "次"),
        ("存货周转天数(天)", "存货周转天数", "天"),
        ("应收账款周转率(次)", "应收账款周转率", "次"),
        ("应收账款周转天数(天)", "应收账款周转天数", "天"),
    ]
    for key, label, suffix in efficiency_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    # 现金流相关
    cf_fields = [
        ("经营现金净流量对销售收入比率(%)", "经营现金流/营收比", "%"),
        ("经营现金净流量与净利润的比率(%)", "经营现金流/净利润比", "%"),
    ]
    for key, label, suffix in cf_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    return "，".join(parts) + "。" if valid else None


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent Chunk 构建")
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--financial_only", action="store_true")
    args = parser.parse_args()

    all_chunks = []

    # 1. 研报元数据chunks（标题+评级+EPS预测）
    report_path = os.path.join(RAW_DIR, "reports", "all_reports.csv")
    if not args.financial_only and os.path.exists(report_path):
        report_chunks = build_report_chunks(report_path)
        all_chunks.extend(report_chunks)
    elif not args.financial_only:
        logger.warning(f"研报元数据不存在: {report_path}，跳过")

    # 2. 研报PDF正文chunks（由 03a_parse_pdfplumber.py 生成）
    pdf_chunk_path = os.path.join(RAW_DIR, "report_parsed", "pdfplumber_all_chunks.jsonl")
    if not args.financial_only and os.path.exists(pdf_chunk_path):
        pdf_chunks = []
        with open(pdf_chunk_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    pdf_chunks.append(json.loads(line))
        all_chunks.extend(pdf_chunks)
        logger.info(f"研报PDF正文chunks: {len(pdf_chunks)} 条")
    elif not args.financial_only:
        logger.warning(f"研报PDF正文chunks不存在: {pdf_chunk_path}，跳过（可运行 03a_parse_pdfplumber.py 生成）")

    # 3. 财务数据chunks
    financial_path = os.path.join(RAW_DIR, "financial", "all_financial.json")
    if not args.report_only and os.path.exists(financial_path):
        financial_chunks = build_financial_chunks(financial_path)
        all_chunks.extend(financial_chunks)
    elif not args.report_only:
        logger.warning(f"财务数据不存在: {financial_path}，跳过")

    # 保存
    if all_chunks:
        # JSONL格式（每行一个chunk）
        output_path = os.path.join(CHUNK_DIR, "all_chunks.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # 统计信息
        source_counts = {}
        for c in all_chunks:
            st = c["metadata"]["source_type"]
            source_counts[st] = source_counts.get(st, 0) + 1

        stats = {
            "total_chunks": len(all_chunks),
            "by_source_type": source_counts,
            "unique_stocks": len(set(c["metadata"]["stock_code"] for c in all_chunks)),
            "avg_text_length": sum(len(c["text"]) for c in all_chunks) / len(all_chunks),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(os.path.join(CHUNK_DIR, "chunk_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("=" * 50)
        logger.info("Chunk 构建完成！")
        logger.info(f"  总chunks: {stats['total_chunks']}")
        for st, cnt in source_counts.items():
            logger.info(f"  {st}: {cnt} 条")
        logger.info(f"  覆盖股票: {stats['unique_stocks']} 只")
        logger.info(f"  平均chunk长度: {stats['avg_text_length']:.0f} 字符")
        logger.info(f"  保存路径: {output_path}")
        logger.info("=" * 50)
        logger.info("下一步: 运行 05_build_index.py 构建 FAISS + BM25 索引")
    else:
        logger.error("没有生成任何chunk，请检查原始数据是否存在")


if __name__ == "__main__":
    main()
