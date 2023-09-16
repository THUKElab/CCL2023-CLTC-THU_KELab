# import re
# from ..common import KEY_SRC, KEY_TGT
#
# zh_left_quotation = "“"
# zh_right_quotation = "”"
# zh_end_dot_punctuation = "。？！"
#
# speech_pattern = re.compile('：“.*?”')
# left_half_speech_pattern = re.compile('：“[^”]*$')
# right_half_speech_pattern = re.compile('^[^“”]*?”')
#
#
# def post_process(text_list):
#     result = [quotation_revision(x) for x in text_list]
#     return result
#
#
# def quotation_revision(text):
#     # 移除开头引号
#     if not text:
#         return text
#     revise = text
#     if revise[0] == zh_right_quotation:
#         revise = revise[1:]
#     speech = speech_pattern.search(revise)
#     if speech is not None:
#         for left_id, right_id in speech.regs:
#             assert revise[right_id - 1] == zh_right_quotation
#             if len(revise) > right_id and revise[right_id] in zh_end_dot_punctuation \
#                     and revise[right_id - 2] not in zh_end_dot_punctuation:
#                 # print("invalid speech", speech)
#                 revise = revise[:right_id - 1] + revise[right_id] + zh_right_quotation + revise[right_id + 1:]
#
#     half_speech = left_half_speech_pattern.search(revise)
#     if half_speech is not None and speech_pattern.search(revise) is None:
#         # print("half_speech", half_speech)
#         for left_id, _ in half_speech.regs:
#             dot_punct_id = left_id + 2
#             while dot_punct_id < len(revise) and revise[dot_punct_id] not in zh_end_dot_punctuation:
#                 dot_punct_id += 1
#             if dot_punct_id == len(revise):
#                 revise += zh_right_quotation
#             else:
#                 revise = revise[:dot_punct_id + 1] + zh_right_quotation + revise[dot_punct_id + 1:]
#     return revise
#
#
# def detect_error_quotation(data):
#     for sample in data:
#         src, tgt = sample[KEY_SRC], sample[KEY_TGT]
#         if left_half_speech_pattern.search(src) is not None:
#             print("Left half speech:", src, "||", tgt)
#         if right_half_speech_pattern.search(src) is not None:
#             print("Right half speech:", src, "||", tgt)
#
#
# # if __name__ == '__main__':
# #     # seq_list =[
# #     #     "抽烟的人经常说：“你们嫌烟者堂堂正正地主张在公共场所不抽副流烟的权利。但是我们抽烟的人也有抽烟的权利。难道你们忽视我们的权利、人权吗？",
# #     #     "我对她说：“如果细菌从伤口侵入的话，或许可能导致伤口化脓，所以你别外出，在家里老实点儿”。否则你懂的。",
# #     #
# #     # ]
# #     # for seq in seq_list:
# #     #     quotation_check(seq)
# #
# #     from metrics.ChERRANT.pipeline import parallel_to_m2, evaluate_m2
# #
# #     _DATA_DIR = "/data/home/yejh/nlp/datasets"
# #     ref_m2_file = f"{_DATA_DIR}/CGEC/mucgec/dev.m2.char"
# #     with open(ref_m2_file, 'r', encoding='utf-8') as f:
# #         ref_m2 = f.read().strip().split("\n\n")
# #
# #     # data_file = "/data/home/yejh/nlp/GEC/models/seq2seq/exps/lh_large_lr1e-5_b32_ner0.05_EV/pred/pred_5.txt"
# #     data_file = "/data/home/yejh/nlp/GEC/models/seq2seq/exps/lh_large_lr1e-5_b32_ner0.05_EV/test_pred.txt"
# #     gec_data = read_data(data_file)
# #
# #     if "test" not in data_file:
# #         # Convert to m2 format
# #         hyp_m2, _ = parallel_to_m2(data=gec_data)
# #         print(hyp_m2[:3])
# #         # Compare pred and ref for evaluation
# #         eval_dict = evaluate_m2(hyp_m2, ref_m2)
# #         print(eval_dict)
# #
# #     detect_error_quotation(gec_data)
# #     print("=================== Revise ==================")
# #     cnt_revision = 0
# #     for sample in gec_data:
# #         src, tgt = sample[KEY_SRC], sample[KEY_TGT]
# #         new_tgt = quotation_revision(tgt)
# #         if tgt != new_tgt:
# #             print(src, "||", tgt, "||", new_tgt)
# #             sample[KEY_TGT] = new_tgt
# #             cnt_revision += 1
# #     print(f"Total {cnt_revision} revisions.")
# #
# #     if "test" in data_file:
# #         write_data(
# #             gec_data,
# #             data_file=os.path.join(os.path.dirname(data_file), "pred_test_revision.txt"),
# #             write_id=True
# #         )
# #
# #     if "test" not in data_file:
# #         # Convert to m2 format
# #         hyp_m2, _ = parallel_to_m2(data=gec_data)
# #         # Compare pred and ref for evaluation
# #         eval_dict = evaluate_m2(hyp_m2, ref_m2)
# #         print(eval_dict)
#
# """
# Seq2Seq
# {'fn': 2852, 'tp': 1137, 'fp': 1091, 'P': 0.5103, 'F0.5': 0.4407, 'R': 0.2850}
# {'fn': 2842, 'tp': 1147, 'fp': 1093, 'P': 0.5121, 'F0.5': 0.4429, 'R': 0.2875}
#
# """
#
# """
# ssh://yejh@10.103.11.151:22/home/yejh/anaconda3/envs/allennlp/bin/python -u /home/yejh/nlp/GEC/data/rule.py
# 07/22/2022 22:32:15 - INFO - utils -   Read 1137 samples from /data/home/yejh/nlp/GEC/models/seq2seq/exps/lh_large_lr1e-5_b32_ner0.05_EV/pred/pred_5.txt.
# Right half speech: ”这样注意的话，肯定对孩子好的影响。 || ”这样注意的话，肯定对孩子好的影响。
# Left half speech: ”我指着那个标志之后说：“是的。 || ”我指着那个标志说：“是的。
# Right half speech: ”我指着那个标志之后说：“是的。 || ”我指着那个标志说：“是的。
# Left half speech: 我母亲曾经跟我说过：“不是绿色食品的再洗也洗不掉农药，所以我们的身体没有好处。 || 我母亲曾经跟我说过：“不是绿色食品的再洗也洗不掉农药，所以对我们的身体没有好处。
# Right half speech: 但现在，怎么戒不了了。”以后我每天看到爸爸抽着烟，我很生气。因为爸爸抽的烟家里满地都是。 || 但现在，怎么也戒不了了。”以后我每天看到爸爸抽烟，我很生气。因为爸爸抽的烟弄得家里满地都是。
# Right half speech: ”他的意见是这样，但是吸烟对个人健康带来很多不好影响，它是个毒品。 || ”他的意见是这样，但是吸烟给个人健康带来很多不好的影响，它是个毒品。
# Right half speech: ”这就是社会上基本原则。这样一来，整个世界会一起提高社会生活素质。 || ”这就是社会上的基本原则。这样一来，整个世界会一起提高社会生活质量。
# Right half speech: ”他跟我说“我想做医生因为我父母都是医生”。 || ”他跟我说“我想做医生，因为我父母都是医生”。
# Right half speech: 如果你不知道将来你应该丢面子”。 || 如果你不知道将来你会丢面子”。
# Left half speech: 不过某个时期，我想这样：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。 || 不过某个时期，我想这样：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。
# Right half speech: 上菜的時候，一般先上涼菜，然后上热菜。上菜来以后，一般先要等最重要的或者年纪最大的客人开始吃，其他人才能跟着吃。后来，主人会鼓励客人们吃到饱，常常会说”多吃点”，”慢慢吃”等等。 || 上菜的时候，一般先上凉菜，然后上热菜。上菜来以后，一般先要等最重要的或者年纪最大的客人开始吃，其他人才能跟着吃。后来，主人会鼓励客人们吃到饱，常常会说”多吃点”，”慢慢吃”等等。
# Right half speech: ”我说“因为呢，人们喜欢流行歌曲，还有很大的吸引力，还有除外流行歌曲以外其它的歌曲没有听过。 || ”我说：“因为呢，人们喜欢流行歌曲，它有很大的吸引力，还有除了流行曲以外其它的歌曲没有听过。
# Right half speech: ”虽然个人力量有限，可是每个人的力量同时团团起来的话，这力量是非常大。 || ”虽然个人力量有限，可是每个人的力量同时团结起来的话，这力量是非常大。
# ”这样注意的话，肯定对孩子好的影响。 || ”这样注意的话，肯定对孩子好的影响。 || 这样注意的话，肯定对孩子好的影响。
# half_speech <re.Match object; span=(8, 13), match='：“是的。'>
# ”我指着那个标志之后说：“是的。 || ”我指着那个标志说：“是的。 || 我指着那个标志说：“是的。”
# half_speech <re.Match object; span=(9, 40), match='：“不是绿色食品的再洗也洗不掉农药，所以对我们的身体没有好处。'>
# 我母亲曾经跟我说过：“不是绿色食品的再洗也洗不掉农药，所以我们的身体没有好处。 || 我母亲曾经跟我说过：“不是绿色食品的再洗也洗不掉农药，所以对我们的身体没有好处。 || 我母亲曾经跟我说过：“不是绿色食品的再洗也洗不掉农药，所以对我们的身体没有好处。”
# ”他的意见是这样，但是吸烟对个人健康带来很多不好影响，它是个毒品。 || ”他的意见是这样，但是吸烟给个人健康带来很多不好的影响，它是个毒品。 || 他的意见是这样，但是吸烟给个人健康带来很多不好的影响，它是个毒品。
# ”这就是社会上基本原则。这样一来，整个世界会一起提高社会生活素质。 || ”这就是社会上的基本原则。这样一来，整个世界会一起提高社会生活质量。 || 这就是社会上的基本原则。这样一来，整个世界会一起提高社会生活质量。
# ”他跟我说“我想做医生因为我父母都是医生”。 || ”他跟我说“我想做医生，因为我父母都是医生”。 || 他跟我说“我想做医生，因为我父母都是医生”。
# half_speech <re.Match object; span=(11, 39), match='：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。'>
# 不过某个时期，我想这样：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。 || 不过某个时期，我想这样：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。 || 不过某个时期，我想这样：“因为我在跑步，所以遭受歹徒袭击的话，他们也必须跑步。”
# half_speech <re.Match object; span=(2, 50), match='：“因为学校里没有女孩子，我们的年龄相当于青春期，就特别关心异性。我常常去别的女子中学找女朋友。'>
# 他说“因为学校里没有女孩子，我们的年龄相当于青春期，就是特别关心异性。我常常去了别的女子中学找女朋友。 || 他说：“因为学校里没有女孩子，我们的年龄相当于青春期，就特别关心异性。我常常去别的女子中学找女朋友。 || 他说：“因为学校里没有女孩子，我们的年龄相当于青春期，就特别关心异性。”我常常去别的女子中学找女朋友。
# half_speech <re.Match object; span=(2, 45), match='：“因为呢，人们喜欢流行歌曲，它有很大的吸引力，还有除了流行曲以外其它的歌曲没有听过。'>
# ”我说“因为呢，人们喜欢流行歌曲，还有很大的吸引力，还有除外流行歌曲以外其它的歌曲没有听过。 || ”我说：“因为呢，人们喜欢流行歌曲，它有很大的吸引力，还有除了流行曲以外其它的歌曲没有听过。 || 我说：“因为呢，人们喜欢流行歌曲，它有很大的吸引力，还有除了流行曲以外其它的歌曲没有听过。”
# ”虽然个人力量有限，可是每个人的力量同时团团起来的话，这力量是非常大。 || ”虽然个人力量有限，可是每个人的力量同时团结起来的话，这力量是非常大。 || 虽然个人力量有限，可是每个人的力量同时团结起来的话，这力量是非常大。
#
# """
