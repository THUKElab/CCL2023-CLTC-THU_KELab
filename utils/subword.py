from typing import List


def subword_align(src_line: str, tgt_line: str) -> List[int]:
    """ 对齐分词
    @param src_line: 原始分词
    @param tgt_line: BPE分词
    Example:
        src_line: Humans have many basic needs
        tgt_line: Hum@@ ans have many basic needs
        return: 0 0 1 2 3 4

    """
    src_line = src_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    tgt_line = tgt_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    src_tokens, tgt_tokens = src_line.rstrip().split(), tgt_line.rstrip().split()

    if len(src_tokens) == 0:
        assert len(tgt_tokens) == 0
        return []

    i, j = 0, 0
    aligned_results = []
    try:
        while j < len(tgt_tokens):
            while tgt_tokens[j].endswith("@@"):
                if src_tokens[i].endswith("@@") and tgt_tokens[j] == "@@":
                    break
                if src_tokens[i] == "@@@@@" and tgt_tokens[j] == "@@@@":
                    break
                aligned_results.append(i)
                j += 1
            aligned_results.append(i)
            i += 1
            j += 1
    except RuntimeError:
        print(src_line, tgt_line)
        print(src_tokens)
        print(tgt_tokens)
    assert len(aligned_results) == len(tgt_tokens)
    assert int(aligned_results[-1]) == len(src_tokens) - 1, f"{src_line}, {tgt_line}"
    return aligned_results
