# Post-process model's outputs for the evaluation dataset YACLC
"""
(1) 不采用包含非中文的编辑
    SRC: 几 天 后 学 生 还 没 道 歉 所 以 校 长 决 定 如 果 学 生 不 能 道 歉 学 生 不 得 不 停 课 ( s u s p e n s i o n ) 。
    REF: 几 天 后 ， 学 生 还 没 道 歉 ， 校 长 决 定 ： 如 果 学 生 不 道 歉 ， 他 将 不 得 不 停 课 ( s u s p e n s i o n ) 。
    HYP: 几 天 后 学 生 还 没 道 歉 ， 所 以 校 长 决 定 如 果 学 生 不 道 歉 学 生 不 得 不 停 课 。

    SRC: ( 你 知 道 嘛 , 我 两 年 前 考 了 H S K , 合 格 了 新 H S K 六 级 . 我 自 己 不 能 相 信 .
    REF: ( 你 知 道 嘛 , 我 两 年 前 考 了 H S K , 考 过 了 新 H S K 六 级 。 我 自 己 简 直 不 敢 相 信 。
    HYP: 你 知 道 嘛 , 我 两 年 前 考 了 h s k , 新 h s k 六 级 合 格 了 . 我 自 己 都 不 能 相 信 。

    SRC: 普 通 " 鯛 焼 き " 大 小 大 概 是 和 掌 一 样 的 。
    REF: 普 通 " 鲷 焼 き " 大 小 大 概 是 和 掌 一 样 。
    TGT: 普 通 的 " 鲷 鱼 烧 " 大 小 大 概 是 和 手 掌 一 样 的 。

"""

import os
import sys
import string

sys.path.append(f"{os.path.dirname(__file__)}/../../")
from data.constants import PUNCT_EN, PUNCT_ZH, PUNCT_END, PUNCT_DOT
from utils.str import remove_space

PUNCT_VALID = set(PUNCT_EN) | set(PUNCT_ZH) | set(PUNCT_END) | set(PUNCT_DOT) | set("・")


def check_valid_tokens(text: str):
    text = remove_space(text.strip())
    for c in text:
        if c not in PUNCT_VALID \
                and not '\u4e00' <= c <= '\u9fff' \
                and c not in string.digits:
            return False
    return True


if __name__ == '__main__':
    from data import M2DataReader

    file_errant = sys.argv[1]
    file_output = sys.argv[2]

    reader = M2DataReader()
    dataset = reader.read(file_errant)

    with open(file_output, "w", encoding="utf-8") as f:
        for sample in dataset.samples:
            src, tgt, edits = sample.source[0], sample.target[0], sample.edits[0][0]
            new_edits = []
            for e in edits:
                src_tokens = "".join(e.src_tokens)
                tgt_tokens = "".join(e.tgt_tokens)
                if check_valid_tokens(src_tokens) and check_valid_tokens(tgt_tokens):
                    new_edits.append(e)
                # else:
                #     print(f"SRC: {src}")
                #     print(f"TGT: {tgt}")
                #     print(e)

            new_src_tokens, _ = reader.convert_edits_to_target(
                src.split(),
                [x.to_m2() for x in new_edits],
            )
            f.write(''.join(new_src_tokens) + "\n")
            # print(f"New TGT: {' '.join(new_src_tokens)}")
            # print()
