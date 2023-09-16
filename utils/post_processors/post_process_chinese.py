import sys
from typing import List


# THRESH_LENGTH = 128
# THRESH_LENGTH = 10000

def post_process(
        srcs: List[str],
        tgts: List[str],
        ids: List,
        THRESH_LENGTH=10000,
):
    total = ids[-1] + 1 if isinstance(ids[0], int) else int(ids[-1].strip()) + 1
    results = ["" for _ in range(total)]
    for src, tgt, idx in zip(srcs, tgts, ids):
        src = src.replace("##", "").replace(" ", "")
        tgt = tgt.replace("##", "").replace(" ", "")
        if len(src) >= THRESH_LENGTH or len(tgt) >= THRESH_LENGTH:
            res = src
        else:
            res = tgt
        res = res.rstrip("\n")
        results[int(idx)] += res
    return results



if __name__ == '__main__':
    input_file = sys.argv[1]
    cor_file = sys.argv[2]
    out_file = sys.argv[3]
    id_file = sys.argv[4]
    threshold = sys.argv[5]

    with open(input_file, "r") as f1:
        with open(cor_file, "r") as f2:
            with open(id_file, "r") as f3:
                src_lines, tgt_lines, id_lines = f1.readlines(), f2.readlines(), f3.readlines()

    post_results = post_process(
        src_lines,
        tgt_lines,
        id_lines,
        THRESH_LENGTH=int(threshold),
    )

    with open(out_file, "w") as o:
        for post in post_results:
            o.write(post + "\n")


# with open(input_file, "r") as f1:
#     with open(cor_file, "r") as f2:
#         with open(out_file, "w") as o:
#             srcs, tgts = f1.readlines(), f2.readlines()
#             res_li = ["" for i in range(6000)]
#             for idx, (src, tgt) in enumerate(zip(srcs, tgts)):
#                 src = src.replace(" ", "")
#                 tgt = tgt.replace(" ", "")
#                 if len(src) >= 128 or len(tgt) >= 128:
#                     res = src
#                 else:
#                     res = tgt
#                 res = res.rstrip("\n")
#                 res_li[int(idx)] += res
#             for res in res_li:
#                 o.write(res + "\n")
