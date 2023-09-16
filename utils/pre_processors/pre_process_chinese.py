from bert import tokenization
from typing import List

from utils import get_logger, remove_space
from utils.split_sent import split_sentence

LOGGER = get_logger(__name__)


def pre_process(src_list: List[str], file_vocab: str):
    """ MuCGEC 中文预处理
        1) 分句
        2) Tokenize
    """
    # Step 1: Split Sentences
    sub_sents_list = []
    for idx, line in enumerate(src_list):
        line = remove_space(line.rstrip("\n"))
        sents = split_sentence(line, flag="zh")
        if len(line) < 64:
            sub_sents_list.append([line])
        else:
            sub_sents_list.append(sents)

    sent_list, ids = [], []
    for idx, sub_sents in enumerate(sub_sents_list):
        sent_list.extend(sub_sents)
        ids.extend([idx] * len(sub_sents))

    # Step 2: Tokenize
    tok_sent_list = []
    tokenizer = tokenization.FullTokenizer(
        vocab_file=file_vocab,
        do_lower_case=False,
    )

    for sent in sent_list:
        line = remove_space(sent.strip())
        line = tokenization.convert_to_unicode(line)
        if not line:
            return ''
        tokens = tokenizer.tokenize(line)
        tok_sent_list.append(' '.join(tokens))

    return tok_sent_list, ids
