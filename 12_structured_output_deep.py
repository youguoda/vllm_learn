"""
结构化输出深度对比：FSM 编译耗时 + schema 复杂度 + 高确定性 schema
用法: NO_PROXY="*" python3 12_structured_output_deep.py [vllm|sglang]
"""
import json
import statistics
import sys
import time

import httpx

PORT = {"vllm": 8000, "sglang": 30000}
MODEL = "Qwen/Qwen2.5-1.5B"

SCHEMAS = {
    "simple": {
        "type": "object",
        "properties": {"defect_type": {"type": "string"},
                       "severity": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["defect_type", "severity"]},
    "medium": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Z]{2}[0-9]{4}$"},
            "type": {"type": "string", "enum": ["剥落", "裂纹", "磨损", "腐蚀", "其他"]},
            "severity": {"type": "integer", "minimum": 1, "maximum": 5},
            "confirmed": {"type": "boolean"}, "notes": {"type": "string"}},
        "required": ["id", "type", "severity", "confirmed"]},
    "complex": {
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "defects": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object", "properties": {
                    "type": {"type": "string", "enum": ["剥落", "裂纹", "磨损", "腐蚀"]},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "location": {"type": "object", "properties": {
                        "component": {"type": "string"}, "position_mm": {"type": "number"}},
                        "required": ["component"]}}, "required": ["type", "severity"]}},
            "need_shutdown": {"type": "boolean"}},
        "required": ["report_id", "defects", "need_shutdown"]},
    "high_det": {
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "pattern": "^SEN-[0-9]{6}-[A-Z]{3}$"},
            "status": {"type": "string", "enum": ["正常", "警告", "严重", "停机"]},
            "value_hz": {"type": "number", "minimum": 0, "maximum": 10000},
            "timestamp": {"type": "string",
                          "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"}},
        "required": ["sensor_id", "status", "value_hz", "timestamp"]},
}

PROMPT = ("检测报告：传感器SEN-000042-ABC检测到轴承外圈出现明显剥落，"
          "振动幅值超标4倍，需立即处理。请生成结构化报告。")


def single(client, base, schema_name):
    schema = SCHEMAS[schema_name]
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256, "temperature": 0.0,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": schema_name, "schema": schema}}})
    dt = (time.perf_counter() - t0) * 1000
    try:
        json.loads(r.json()["choices"][0]["message"]["content"])
        valid = True
    except Exception:
        valid = False
    return dt, valid


def main():
    fw = sys.argv[1] if len(sys.argv) > 1 else "vllm"
    base = f"http://127.0.0.1:{PORT[fw]}"
    client = httpx.Client(trust_env=False, timeout=120)
    print(f"\n框架: {fw}")
    print(f"{'Schema':<10} {'第1次(ms)':>10} {'第2次(ms)':>10} {'热身后(ms)':>10} {'合法率':>7}")
    print("-" * 55)
    for sname in ("simple", "medium", "complex", "high_det"):
        times, valids = [], []
        for i in range(5):
            t, v = single(client, base, sname)
            times.append(t); valids.append(v)
        legal = sum(valids) / len(valids)
        print(f"{sname:<10} {times[0]:>10.0f} {times[1]:>10.0f} "
              f"{statistics.median(times[2:]):>10.0f} {legal:>6.0%}")


if __name__ == "__main__":
    main()
