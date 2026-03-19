import os
import json
import time
import argparse
import logging
import random
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置 ============
PDF_DIR = "./data/raw/report_pdfs"
RESULT_FILE = "./data/raw/report_parsed/marker_all_results.json"

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Marker PDF 提取")
    parser.add_argument("--max_pdfs", type=int, default=0, help="最多处理几篇（0=全部）")
    parser.add_argument("--sample", action="store_true", help="随机抽样而非按顺序")
    args = parser.parse_args()

    # 加载模型（首次会自动下载）
    logger.info("加载 Marker 模型...")
    model_start = time.time()
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    logger.info(f"模型加载完成，耗时 {time.time() - model_start:.1f}秒")

    # 获取所有 PDF
    all_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    logger.info(f"共 {len(all_pdfs)} 篇 PDF")

    # 加载已完成的结果（断点续传）
    results = []
    done_files = set()
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
            done_files = {r['file'] for r in results}
        logger.info(f"已完成 {len(done_files)} 篇，跳过")

    # 过滤已完成的
    remaining = [f for f in all_pdfs if f not in done_files]

    # 随机抽样
    if args.sample and args.max_pdfs > 0:
        remaining = random.sample(remaining, min(args.max_pdfs, len(remaining)))
    elif args.max_pdfs > 0:
        remaining = remaining[:args.max_pdfs]

    logger.info(f"本次处理: {len(remaining)} 篇")

    # 处理
    total_start = time.time()
    success = 0
    fail = 0
    for i, pdf_file in enumerate(remaining):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        start = time.time()

        try:
            result = converter(pdf_path)
            text = result.markdown
            elapsed = time.time() - start

            results.append({"file": pdf_file, "text": text, "time": elapsed})
            success += 1

            total_elapsed = time.time() - total_start
            avg = total_elapsed / (i + 1)
            remain_min = avg * (len(remaining) - i - 1) / 60
            logger.info(f"[{i+1}/{len(remaining)}] {elapsed:.1f}秒 | "
                       f"{len(text)}字符 | 剩余{remain_min:.0f}分钟 | {pdf_file}")
        except Exception as e:
            fail += 1
            logger.warning(f"[{i+1}/{len(remaining)}] 失败 | {pdf_file} | {e}")

        # 每 50 篇保存 checkpoint
        if (i + 1) % 50 == 0:
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"  [checkpoint] 已保存 {len(results)} 篇 | 成功{success} 失败{fail}")

    # 最终保存
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计
    if results:
        times = [r['time'] for r in results]
        texts = [len(r['text']) for r in results]
        logger.info("=" * 60)
        logger.info(f"完成! 成功{success}篇, 失败{fail}篇")
        logger.info(f"平均耗时: {sum(times)/len(times):.1f}秒/篇")
        logger.info(f"总耗时: {sum(times)/3600:.1f}小时")
        logger.info(f"平均字符数: {sum(texts)/len(texts):.0f}")
        logger.info(f"保存到: {RESULT_FILE}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
