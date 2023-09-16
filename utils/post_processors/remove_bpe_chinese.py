import sys
from tqdm import tqdm
from multiprocessing import Pool


def remove_bpe(line):
    line = line.strip().replace("##", "").replace(" ", "")
    return line


with Pool(64) as pool:
    for ret in pool.imap(remove_bpe, tqdm(sys.stdin), chunksize=1024):
        print(ret)
