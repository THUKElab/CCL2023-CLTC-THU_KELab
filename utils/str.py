import re
from typing import List, Union

import spacy
from blingfire import text_to_sentences
from nltk import sent_tokenize, TreebankWordDetokenizer
from opencc import OpenCC

SIMPLIFIER = OpenCC("t2s")
SPACY_MODEL = spacy.load('en_core_web_sm')
detokenizer = TreebankWordDetokenizer()


def remove_space(batch: Union[str, List[str]]):
    def _remove_space(text: str):
        text = text.strip().replace("\u3000", " ").replace("\xa0", " ")
        text = "".join(text.split())
        return text

    if isinstance(batch, str):
        return _remove_space(batch)
    else:
        return [_remove_space(x) for x in batch]


def tokenize(text: str, no_space: bool = False):
    if no_space:
        text = remove_space(text)
    doc = SPACY_MODEL(text.strip(), disable=['parser', 'tagger', 'ner'])
    tokens = [str(token) for token in doc]
    return tokens


def tokenize_batch(text_list: List[str], no_space: bool = False):
    if no_space:
        text_list = [remove_space(x) for x in text_list]
    docs = SPACY_MODEL.pipe(text_list, batch_size=1024, disable=['parser', 'tagger', 'ner'])
    docs = [[x.text for x in line] for line in docs]
    return docs


def detokenize(sent: str):
    def handle_double_quote(sent):
        cur_str = ''
        exp_left = True
        ignore_space = False
        for char in sent:
            if char == '"':
                if exp_left:  # this is a left "
                    cur_str = cur_str.rstrip() + ' "'
                    exp_left = (not exp_left)
                    ignore_space = True
                else:  # this is a right "
                    cur_str = cur_str.rstrip() + '" '
                    exp_left = (not exp_left)
                    ignore_space = False
            else:
                if ignore_space:  # expecting right
                    if char == ' ':
                        continue
                    else:
                        cur_str = cur_str + char
                        ignore_space = False
                else:
                    cur_str = cur_str + char
        cur_str = cur_str.strip()
        cur_str = re.sub(r'[ ]+', ' ', cur_str)
        return cur_str

    def postprocess_space(sent):
        sent = re.sub(r'[ ]+\.', '.', sent)
        sent = re.sub(r'[ ]+,', ',', sent)
        sent = re.sub(r'[ ]+!', '!', sent)
        sent = re.sub(r'[ ]+\?', '?', sent)
        sent = re.sub(r'\([ ]+', '(', sent)
        sent = re.sub(r'[ ]+\)', ')', sent)
        sent = re.sub(r' \'s( |\.|,|!|\?)', r"'s\1", sent)
        sent = re.sub(r'n \'t( |\.|,|!|\?)', r"n't\1", sent)
        return sent

    # Clean raw sent
    sent = re.sub(r'\' s ', '\'s ', sent)
    toks = sent.split()
    if len([1 for t in toks if t == "'"]) % 2 == 0:
        toks = ['"' if t == "'" else t for t in toks]
    sent = ' '.join(toks)

    sents = sent_tokenize(sent)
    final_sents = []
    for _sent in sents:
        _sent = detokenizer.detokenize(_sent.split())
        res = handle_double_quote(_sent)
        if res == -1:
            print('unbalanced double quote')
            print(_sent)
        else:
            _sent = res
        final_sents.append(_sent)
    sent = ' '.join(final_sents)
    sent = postprocess_space(sent)
    return sent


def simplify_chinese(text: str):
    return SIMPLIFIER.convert(text)


def check_all_chinese(text: str):
    """ 判断字符串是否全部由中文组成
        1) 空格、字母不是中文
        2) 日文、韩文不是中文
    """
    return all(['\u4e00' <= ch <= '\u9fff' for ch in text])


def split_sentence(text: str):
    sentences = text_to_sentences(text.strip()).split('\n')
    return sentences

# def split_word_helper(sents: List[str], word_level=False):
#     split_sents, vocab_dict = [], Counter()
#     for sent in sents:
#         if word_level:
#             split_sent = list(jieba.cut(sent))
#         else:
#             split_sent = char_pattern.findall(sent)
#         split_sents.append(split_sent)
#         vocab_dict.update(split_sent)
#     return split_sents, vocab_dict
#
#
# def split_word(sents: List[str], word_level=False, num_worker=1):
#     if num_worker == 1:
#         return split_word_helper(sents, word_level=word_level)
#     results = multiprocess_helper(split_word_helper, (sents, word_level), num_worker=num_worker)
#     total_seg_sents, total_word_vocab_dict = [], Counter()
#     for seg, vocab in results:
#         total_seg_sents.extend(seg)
#         total_word_vocab_dict += vocab
#     return total_seg_sents, total_word_vocab_dict
#
#
# def multiprocess_helper(func, args, num_worker=1):
#     returns, results = [], []
#     split_input = args[0]
#     step = math.ceil(len(split_input) / num_worker)
#     pool = Pool(processes=num_worker)
#     for i in range(0, len(split_input), step):
#         results.append(pool.apply_async(func, (split_input[i:i + step], *args[1:])))
#     pool.close()
#     pool.join()
#     for res in results:
#         returns.append(res.get())
#     return returns
#
#
# def split_sentence(document: str, lang: str = "all", strategy: str = "all", split_length: int = 60, limit: int = 510):
#     """
#     Args:
#         document: 文档
#         lang: "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
#         strategy: 分句策略
#             all: 划分每个句子
#             greedy: 划分结果可能包含多个句子，但每个部分长度不超过 split_length
#         split_length:
#         limit: 默认单句最大长度为510个字符
#     Returns: Type:list
#     """
#     assert strategy in ['all', 'greedy']
#     sent_list = []
#     try:
#         if lang == "zh":
#             # 中文单字符断句符
#             document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             # 特殊引号
#             document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)
#         elif lang == "en":
#             # 英文单字符断句符
#             document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             # 特殊引号
#             document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)
#         else:
#             document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n', document)
#
#         sent_list_ori = document.splitlines()
#         for sent in sent_list_ori:
#             sent = sent.strip()
#             while len(sent) > limit:
#                 stop = limit - 1
#                 while stop >= 0 and sent[stop] not in ",;，；":
#                     stop -= 1
#                 if stop < 0:
#                     stop = limit - 1
#                 sent_list.append(sent[:stop + 1])
#                 sent = sent[stop + 1:]
#             if sent:
#                 sent_list.append(sent)
#     except RuntimeError and IndexError:
#         print(f"Fail to split document: {document}")
#         sent_list.clear()
#         sent_list.append(document)
#
#     if strategy == 'all':
#         return sent_list
#     elif strategy == 'greedy':
#         cur_sent = ""
#         merge_sent_list = []
#         for sent in sent_list:
#             if len(sent) + len(cur_sent) <= split_length:
#                 cur_sent += sent
#             else:
#                 if cur_sent:
#                     merge_sent_list.append(cur_sent)
#                 cur_sent = sent
#         if cur_sent:
#             merge_sent_list.append(cur_sent)
#         return merge_sent_list
#     else:
#         return sent_list
