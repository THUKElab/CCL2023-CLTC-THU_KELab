from dataclasses import dataclass, field
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)


@dataclass
class LabelSmoothedCrossEntropyCriterionAugmentConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    regularization_weight: float = field(
        default=1.0,
        metadata={"help": "weight for the regularization loss"}
    )


@register_criterion(
    "augmented_label_smoothed_cross_entropy",
    dataclass=LabelSmoothedCrossEntropyCriterionAugmentConfig,
)
class AugmentedLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            regularization_weight=1.0,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.regularization_weight = regularization_weight

    def forward(self, model, sample, reduce=True):
        """ Compute the loss for the given sample.
            Returns a tuple with three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training
        """
        if 'primary' not in sample or 'secondary' not in sample:
            return super().forward(model, sample, reduce=reduce)

        # 1. Compute cross entropy loss of primary samples
        primary_net_output = model(**sample['primary']['net_input'])
        primary_loss, primary_nll_loss = self.compute_loss(
            model,
            primary_net_output,
            sample['primary'],
            reduce=reduce,
        )
        primary_sample_size = (
            sample['primary']["target"].size(0) if self.sentence_avg
            else sample['primary']["ntokens"]
        )

        # 2. Compute cross entropy loss of secondary samples
        secondary_net_output = model(**sample['secondary']['net_input'])
        secondary_loss, secondary_nll_loss = self.compute_loss(
            model,
            secondary_net_output,
            sample['secondary'],
            reduce=reduce,
        )
        secondary_sample_size = (
            sample['secondary']['target'].size(0) if self.sentence_avg
            else sample['secondary']['ntokens']
        )

        loss = primary_loss + secondary_loss
        nll_loss = primary_nll_loss + secondary_nll_loss
        ntokens = sample['primary']['ntokens'] + sample['secondary']['ntokens']
        nsentences = sample['primary']['target'].size(0) + sample['secondary']['target'].size(0)
        sample_size = primary_sample_size + secondary_sample_size
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        # print(logging_output)

        if abs(self.regularization_weight) >= 1e-9:
            # 3. Compute regularization loss of primary samples and secondary samples
            primary_targets = model.get_targets(sample['primary'], primary_net_output).unsqueeze(-1)
            secondary_targets = model.get_targets(sample['secondary'], secondary_net_output).unsqueeze(-1)
            pad_mask = primary_targets.eq(self.padding_idx) | secondary_targets.eq(self.padding_idx)
            regularization_loss = self.compute_regularization_loss(
                model,
                primary_net_output,
                secondary_net_output,
                pad_mask=pad_mask,
                reduce=reduce,
            )
            loss += self.regularization_weight * regularization_loss
            logging_output["regularization_loss"] = regularization_loss.data

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, primary_net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def compute_regularization_loss(
            cls,
            model,
            primary_net_output,
            secondary_net_output,
            pad_mask=None,
            reduce=True,
    ):
        mean_net_output = (primary_net_output[0] + secondary_net_output[0]) / 2
        m = model.get_normalized_probs((mean_net_output,), log_probs=False) + 1e-10
        p = model.get_normalized_probs(primary_net_output, log_probs=True) + 1e-10
        q = model.get_normalized_probs(secondary_net_output, log_probs=True) + 1e-10

        primary_loss = torch.nn.functional.kl_div(p, m, reduction='none')
        secondary_loss = torch.nn.functional.kl_div(q, m, reduction='none')
        if pad_mask is not None:
            primary_loss.masked_fill_(pad_mask, 0.)
            secondary_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            primary_loss = primary_loss.sum()
            secondary_loss = secondary_loss.sum()

        loss = (primary_loss + secondary_loss) / 2
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        regularization_loss_sum = utils.item(sum(log.get('regularization_loss', 0) for log in logging_outputs))
        metrics.log_scalar('regularization_loss', regularization_loss_sum / sample_size, sample_size, round=3)
