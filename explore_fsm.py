"""
explore_fsm.py — 探索"正则→FSM→合法字符"这条约束解码的核心链路

用 interegular(vLLM/outlines 依赖之一) 把一个简化的 JSON 正则编译成 FSM,
看状态数、初始状态下的合法首字符——理解为什么首次请求慢(编译 FSM)。
"""
import interegular


def main():
    # 简化的 JSON object 正则: {"defect_type":"...","severity":N}
    # 真实 schema 正则更长, 这里用简化版演示原理
    regex = r'\{"defect_type":"[a-z ]+","severity":[1-5]\}'

    print("=" * 60)
    print("  正则 → FSM → 合法字符 探索")
    print("=" * 60)
    print(f"\n正则: {regex}")

    pattern = interegular.parse_pattern(regex)
    fsm = pattern.to_fsm()

    print(f"\nFSM 状态数: {len(fsm.states)}")
    print(f"初始状态: {fsm.initial}")
    print(f"接受状态数: {len(fsm.finals)}")

    # 初始状态下哪些字符合法
    init = fsm.initial
    alphabet = fsm.alphabet
    valid_first = []
    for sym in alphabet:
        if sym == interegular.fsm.anything_else:
            continue
        dest = fsm.map.get(init, {}).get(alphabet[sym])
        if dest is not None:
            valid_first.append(sym)
    print(f"\n初始状态合法首字符: {valid_first}  (应只有 '{{')")

    # 模拟走几步: { → " → ...
    print("\n模拟逐字符走 FSM (每步只有合法字符能走):")
    state = init
    for ch in '{"defect':
        idx = alphabet.get(ch, alphabet[interegular.fsm.anything_else])
        nxt = fsm.map.get(state, {}).get(idx)
        # 列出当前状态合法字符数
        valid_n = sum(1 for s in alphabet if s != interegular.fsm.anything_else
                      and fsm.map.get(state, {}).get(alphabet[s]) is not None)
        print(f"  状态 {state:>2} --'{ch}'--> {nxt}   (此状态合法字符数: {valid_n})")
        state = nxt

    print("\n核心洞察:")
    print("  - schema 越复杂, 正则越长, FSM 状态越多 → 首次编译越慢")
    print("  - 每步 decode: 用当前状态查'合法 token 集合'(bitmask), 非法 token logit 置 -inf")
    print("  - 确定性段(如 \"defect_type\":\" 只有一条路) → SGLang Compressed FSM 一步跳过")


if __name__ == "__main__":
    main()
