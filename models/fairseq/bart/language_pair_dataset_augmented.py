# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data import LanguagePairDataset, data_utils

from utils import get_logger

logger = get_logger(__name__)


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        pad_to_multiple=1,
        augmenter=None,
        src_dict=None,
        corrupt_target: bool = False,
):
    """ Build Augmented Samples
        sample = {
            'primary': {
                'id': torch.tensor,
                'nsentences': int,
                'ntokens': int,
                'net_input': {
                    'src_tokens': torch.tensor,
                    'src_lengths': torch.tensor,
                    'prev_output_tokens': torch.tensor,
                },
                'target': torch.tensor,
            },
            'secondary': {
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
        }
    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    # def check_alignment(alignment, src_len, tgt_len):
    #     if alignment is None or len(alignment) == 0:
    #         return False
    #     if (
    #             alignment[:, 0].max().item() >= src_len - 1
    #             or alignment[:, 1].max().item() >= tgt_len - 1
    #     ):
    #         logger.warning("alignment size mismatch found, skipping alignment!")
    #         return False
    #     return True
    #
    # def compute_alignment_weights(alignments):
    #     """
    #     Given a tensor of shape [:, 2] containing the source-target indices
    #     corresponding to the alignments, a weight vector containing the
    #     inverse frequency of each target index is computed.
    #     For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
    #     a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
    #     index 3 is repeated twice)
    #     """
    #     align_tgt = alignments[:, 1]
    #     _, align_tgt_i, align_tgt_c = torch.unique(
    #         align_tgt, return_inverse=True, return_counts=True
    #     )
    #     align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
    #     return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    # MixEdit: To get augmented src_tokens and src_lengths
    augmented_src_tokens_orig, augmented_src_tokens, augmented_src_lengths = None, None, None
    if augmenter is not None:
        assert src_dict is not None
        augmented_src_tokens_orig = [
            augmenter.augment_sample_for_fairseq(
                x, src_dict,
            ) for x in [s["target"] for s in samples]
        ]
        augmented_src_tokens = data_utils.collate_tokens(
            augmented_src_tokens_orig,
            pad_idx,
            eos_idx,
            left_pad=left_pad_source,
            move_eos_to_beginning=False,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )
        augmented_src_tokens = augmented_src_tokens.index_select(0, sort_order)
        augmented_src_lengths = torch.LongTensor(
            [x.ne(pad_idx).long().sum() for x in augmented_src_tokens_orig]
        ).index_select(0, sort_order)

    target, prev_output_tokens, augmented_prev_output_tokens = None, None, None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )

        if corrupt_target:
            # Apply MixEdit to prev_output_tokens
            augmented_prev_output_tokens = data_utils.collate_tokens(
                augmented_src_tokens_orig,
                pad_idx,
                eos_idx,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
                pad_to_multiple=pad_to_multiple,
            )
        else:
            augmented_prev_output_tokens = prev_output_tokens.clone()

    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "primary": {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "target": target,
        },
        "secondary": {
            "id": id.clone(),
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": augmented_src_tokens,
                "src_lengths": augmented_src_lengths,
            },
            "target": target.clone(),
        },
    }

    if prev_output_tokens is not None:
        batch["primary"]["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)

    if augmented_prev_output_tokens is not None:
        batch["secondary"]["net_input"]["prev_output_tokens"] \
            = augmented_prev_output_tokens.index_select(0, sort_order)
    else:
        batch["secondary"]["net_input"]["prev_output_tokens"] \
            = batch["primary"]["net_input"]["prev_output_tokens"].clone()


    # if samples[0].get("alignment", None) is not None:
    #     bsz, tgt_sz = batch["target"].shape
    #     src_sz = batch["net_input"]["src_tokens"].shape[1]
    #
    #     offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
    #     offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
    #     if left_pad_source:
    #         offsets[:, 0] += src_sz - src_lengths
    #     if left_pad_target:
    #         offsets[:, 1] += tgt_sz - tgt_lengths
    #
    #     alignments = [
    #         alignment + offset
    #         for align_idx, offset, src_len, tgt_len in zip(
    #             sort_order, offsets, src_lengths, tgt_lengths
    #         )
    #         for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
    #         if check_alignment(alignment, src_len, tgt_len)
    #     ]
    #
    #     if len(alignments) > 0:
    #         alignments = torch.cat(alignments, dim=0)
    #         align_weights = compute_alignment_weights(alignments)
    #
    #         batch["alignments"] = alignments
    #         batch["align_weights"] = align_weights
    #
    # if samples[0].get("constraints", None) is not None:
    #     # Collate the packed constraints across the samples, padding to
    #     # the length of the longest sample.
    #     lens = [sample.get("constraints").size(0) for sample in samples]
    #     max_len = max(lens)
    #     constraints = torch.zeros((len(samples), max(lens))).long()
    #     for i, sample in enumerate(samples):
    #         constraints[i, 0: lens[i]] = samples[i].get("constraints")
    #     batch["constraints"] = constraints.index_select(0, sort_order)

    return batch


class AugmentedLanguagePairDataset(LanguagePairDataset):
    """ an augmented pair of torch.utils.data.Datasets. """

    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            augmenter=None,
            corrupt_target=True,
    ):
        super().__init__(
            src,
            src_sizes,
            src_dict,
            tgt=tgt,
            tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            shuffle=shuffle,
            input_feeding=input_feeding,
            remove_eos_from_source=remove_eos_from_source,
            append_eos_to_target=append_eos_to_target,
            align_dataset=align_dataset,
            constraints=constraints,
            append_bos=append_bos,
            eos=eos,
            num_buckets=num_buckets,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
            pad_to_multiple=pad_to_multiple,
        )
        self.augmenter = augmenter
        self.corrupt_target=corrupt_target

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            augmenter=self.augmenter,
            src_dict=self.src_dict,
            corrupt_target=self.corrupt_target,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res
