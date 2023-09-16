# This file customizes some specific implementation for GEC task
import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from utils import get_logger
from .language_pair_dataset_augmented import AugmentedLanguagePairDataset

LOGGER = get_logger(__name__)


def load_augmented_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        augmenter=None,
        corrupt_target=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        LOGGER.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        LOGGER.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return AugmentedLanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        augmenter=augmenter,
        corrupt_target=corrupt_target,
    )


@dataclass
class GECConfig(TranslationConfig):
    # options for task-specific data augmentation
    augmentation_schema: str = field(
        default="none",
        metadata={
            "help": "augmentation schema: e.g. `cut_off`, `src_cut_off`, `trg_cut_off`",
            "choices": ["none", "cut_off", "src_cut_off", "trg_cut_off", "copy"],
        }
    )
    augmentation_merge_sample: bool = field(
        default=False,
        metadata={"help": "merge original and augmented samples together"}
    )
    augmentation_masking_schema: str = field(
        default="word",
        metadata={
            "help": "augmentation masking schema: e.g. `word`, `span`",
            "choices": ["word", "span"],
        }
    )
    augmentation_masking_probability: float = field(
        default=0.15,
        metadata={
            "help": "augmentation masking probability",
        }
    )
    augmentation_replacing_schema: str = field(
        default="mask",
        metadata={
            "help": "augmentation replacing schema: e.g. `mask`, `random`, `mixed`",
            "choices": ["mask", "random", "mixed"],
        }
    )
    augmentation_span_type: str = field(
        default="sample",
        metadata={
            "help": "augmentation span type e.g. sample, w_sample, ws_sample, etc.",
            "choices": ["sample", "w_sample", "ws_sample"],
        }
    )
    augmentation_span_len_dist: str = field(
        default="geometric",
        metadata={"help": "augmentation span length distribution e.g. geometric, poisson, etc."}
    )
    augmentation_max_span_len: int = field(
        default=10,
        metadata={"help": "augmentation maximum span length"}
    )
    augmentation_min_num_spans: int = field(
        default=5,
        metadata={"help": "augmentation minimum number of spans"}
    )
    augmentation_geometric_prob: float = field(
        default=0.2,
        metadata={"help": "augmentation minimum number of spans"}
    )
    augmentation_poisson_lambda: float = field(
        default=5.0,
        metadata={"help": "augmentation lambda of poisson distribution"}
    )


@register_task("gec", dataclass=GECConfig)
class GECTask(TranslationTask):
    def __init__(self, cfg: GECConfig, src_dict, tgt_dict, score_key="F"):
        super().__init__(cfg, src_dict, tgt_dict)
        self.cfg: GECConfig = cfg
        self.score_key = score_key
        self.cfg_all = None
        self.trainer = None
        self.bpe = None
        self.tokenizer = None
        self.metric = None
        self.eval_data = None
        self.eval_input = None
        self.sequence_generator = None
        self.model_score = []
        self.augmenter = None

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False,
    ):
        sample = self.augment_sample(sample)
        return super().train_step(
            sample=sample,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            update_num=update_num,
            ignore_grad=ignore_grad,
        )

    def augment_sample(self, sample):
        if self.cfg.augmentation_schema in ["none"]:
            return sample

        if "secondary" in sample:
            augmented_sample = sample["secondary"]
        else:
            augmented_sample = {
                'id': sample['id'].clone(),
                'nsentences': sample['nsentences'],
                'ntokens': sample['ntokens'],
                'net_input': {
                    'src_tokens': None,
                    'src_lengths': sample['net_input']['src_lengths'].clone(),
                    'prev_output_tokens': None,
                },
                'target': sample['target'].clone()
            }

        if self.cfg.augmentation_schema == 'cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['src_tokens']
                if "secondary" in sample else sample['net_input']['src_tokens'],
                self.src_dict,
            )
            # 使用 src_dict 而不是 tgt_dict，因为当使用 transformer 模型时会出现未知原因导致 tgt_dict 扩容
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['prev_output_tokens']
                if "secondary" in sample else sample['net_input']['prev_output_tokens'],
                self.src_dict,
            )
        elif self.cfg.augmentation_schema == 'src_cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['src_tokens']
                if "secondary" in sample else sample['net_input']['src_tokens'],
                self.src_dict,
            )
            if not "secondary" in sample:
                augmented_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].clone()
        elif self.cfg.augmentation_schema == 'trg_cut_off':
            if not "secondary" in sample:
                augmented_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].clone()
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['prev_output_tokens']
                if "secondary" in sample else sample['net_input']['prev_output_tokens'],
                self.src_dict,
            )
        elif self.cfg.augmentation_schema == "copy":
            # Just for debug
            augmented_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].clone()
            augmented_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].clone()
        else:
            raise ValueError("Augmentation schema {0} is not supported".format(self.cfg.augmentation_schema))

        if self.cfg.augmentation_merge_sample:
            sample = {
                'id': torch.cat((sample['id'], augmented_sample['id']), dim=0),
                'nsentences': sample['nsentences'] + augmented_sample['nsentences'],
                'ntokens': sample['ntokens'] + augmented_sample['ntokens'],
                'net_input': {
                    'src_tokens': torch.cat(
                        [
                            sample['net_input']['src_tokens'],
                            augmented_sample['net_input']['src_tokens'],
                        ], dim=0,
                    ),
                    'src_lengths': torch.cat(
                        [
                            sample['net_input']['src_lengths'],
                            augmented_sample['net_input']['src_lengths'],
                        ], dim=0,
                    ),
                    'prev_output_tokens': torch.cat(
                        [
                            sample['net_input']['prev_output_tokens'],
                            augmented_sample['net_input']['prev_output_tokens'],
                        ], dim=0
                    ),
                },
                'target': torch.cat(
                    [
                        sample['target'],
                        augmented_sample['target'],
                    ], dim=0,
                )
            }
        elif "secondary" not in sample:
            sample = {
                'primary': sample,
                'secondary': augmented_sample,
            }
        return sample

    def _mask_tokens(self, inputs, vocab_dict):
        if self.cfg.augmentation_masking_schema == 'word':
            masked_inputs = self._mask_tokens_by_word(inputs, vocab_dict)
        elif self.cfg.augmentation_masking_schema == 'span':
            masked_inputs = self._mask_tokens_by_span(inputs, vocab_dict)
        else:
            raise ValueError("The masking schema {0} is not supported".format(self.cfg.augmentation_masking_schema))
        return masked_inputs

    def _mask_tokens_by_word(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) \
                                  & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.bernoulli(torch.full(
            inputs.shape,
            self.cfg.augmentation_masking_probability,
            device=inputs.device,
        )).bool()

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _mask_tokens_by_span(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        span_info_list = self._generate_spans(inputs)

        num_spans = len(span_info_list)
        masked_span_list = np.random.binomial(
            1,
            self.cfg.augmentation_masking_probability,
            size=num_spans,
        ).astype(bool)
        masked_span_list = [span_info_list[i] for i, masked in enumerate(masked_span_list) if masked]

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) \
                                  & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.zeros_like(inputs)
        for batch_index, seq_index, span_length in masked_span_list:
            random_masking_indices[batch_index, seq_index:seq_index + span_length] = 1

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices.bool() & available_token_indices
        self._replace_token(
            masked_inputs,
            masking_indices,
            unk_index,
            vocab_size,
        )
        return masked_inputs

    def _sample_span_length(self, span_len_dist, max_span_len, geometric_prob=0.2, poisson_lambda=5.0):
        if span_len_dist == 'geometric':
            span_length = min(np.random.geometric(geometric_prob) + 1, max_span_len)
        elif span_len_dist == 'poisson':
            span_length = min(np.random.poisson(poisson_lambda) + 1, max_span_len)
        else:
            span_length = np.random.randint(max_span_len) + 1
        return span_length

    def _get_default_spans(self, batch_index, seq_length, num_spans):
        span_length = int((seq_length - 2) / num_spans)
        last_span_length = seq_length - 2 - (num_spans - 1) * span_length
        span_infos = []
        for i in range(num_spans):
            span_info = (batch_index, 1 + i * span_length, span_length if i < num_spans - 1 else last_span_length)
            span_infos.append(span_info)

        return span_infos

    def _generate_spans(self, inputs):
        if self.cfg.augmentation_span_type == 'sample':
            span_info_list = self._generate_spans_by_sample(inputs)
        elif self.cfg.augmentation_span_type == 'w_sample':
            span_info_list = self._generate_spans_by_w_sample(inputs)
        elif self.cfg.augmentation_span_type == 'ws_sample':
            span_info_list = self._generate_spans_by_ws_sample(inputs)
        else:
            raise ValueError("Span type {0} is not supported".format(self.cfg.augmentation_span_type))

        return span_info_list

    def _generate_spans_by_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index <= max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)
                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _generate_spans_by_w_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[0] - seq_index).item() if word_index.nelement() > 0 \
                        else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _generate_spans_by_ws_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_segments = (inputs >> 25) - 1
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)

                segment_start_id = input_segments[batch_index, seq_index]
                segment_end_id = input_segments[batch_index, seq_index + span_length - 1]
                if segment_start_id != segment_end_id:
                    segment_index = (input_segments[batch_index, :] == segment_start_id).nonzero().squeeze(-1)
                    span_length = (segment_index[-1] - seq_index + 1).item()

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[0] - seq_index).item() if word_index.nelement() > 0 \
                        else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _replace_token(self, inputs, masking_indices, mask_index, vocab_size):
        if self.cfg.augmentation_replacing_schema == 'mask':
            inputs[masking_indices] = mask_index
        elif self.cfg.augmentation_replacing_schema == 'random':
            random_words = torch.randint(
                vocab_size,
                inputs.shape,
                device=inputs.device,
                dtype=torch.long,
            )
            inputs[masking_indices] = random_words[masking_indices]
        elif self.cfg.augmentation_replacing_schema == 'mixed':
            # 80% of the time, we replace masked input tokens with <unk> token
            mask_token_indices = torch.bernoulli(torch.full(
                inputs.shape,
                0.8,
                device=inputs.device,
            )).bool() & masking_indices
            inputs[mask_token_indices] = mask_index

            # 10% of the time, we replace masked input tokens with random word
            random_token_indices = torch.bernoulli(torch.full(
                inputs.shape,
                0.5,
                device=inputs.device,
            )).bool() & masking_indices & ~mask_token_indices
            random_words = torch.randint(
                vocab_size,
                inputs.shape,
                device=inputs.device,
                dtype=torch.long,
            )
            inputs[random_token_indices] = random_words[random_token_indices]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            raise ValueError(
                "The replacing schema: {0} is not supported. "
                "Only support ['mask', 'random', 'mixed']".format(self.cfg.augmentation_replacing_schema)
            )
