from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.distributions as D

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from allennlp.training.metrics import Average


@Model.register("bernoulli_gen_enc_classifier")
class BernoulliGenEncClassifier(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        generator: Params,
        encoder: Params,
        samples: int,
        reg_loss_lambda: float,
        desired_length: float,
        reg_loss_mu: float,
        rationale_extractor: Model = None,
        supervise_rationale: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(BernoulliGenEncClassifier, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")

        self._generator = Model.from_params(
            vocab=vocab,
            regularizer=regularizer,
            initializer=initializer,
            params=Params(generator),
            supervise_rationale=supervise_rationale,
            max_length_ratio=desired_length,
        )
        self._encoder = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(encoder),
        )

        self._samples = samples
        self._reg_loss_lambda = reg_loss_lambda
        self._reg_loss_mu = reg_loss_mu
        self._desired_length = min(1.0, max(0.0, desired_length))
        self._rationale_extractor = rationale_extractor

        self._loss_tracks = {
            k: Average()
            for k in [
                "_lasso_loss",
                "_base_loss",
                "_rat_length",
                "_fused_lasso_loss",
                "_censored_lasso_loss",
                "_generator_loss",
            ]
        }

        self._supervise_rationale = supervise_rationale

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, rationale=None) -> Dict[str, Any]:
        # pylint: disable=arguments-differ

        generator_dict = self._generator(document, query, label, rationale=rationale)
        mask = generator_dict["mask"]
        assert "probs" in generator_dict

        prob_z = generator_dict["probs"]

        assert len(prob_z.shape) == 2

        output_dict = {}

        sampler = D.bernoulli.Bernoulli(probs=prob_z)
        if self.prediction_mode or not self.training:
            if self._rationale_extractor is None:
                sample_z = generator_dict["predicted_rationale"].float()
            else:
                sample_z = self._rationale_extractor.extract_rationale(prob_z, document, as_one_hot=True)
                output_dict["rationale"] = self._rationale_extractor.extract_rationale(
                    prob_z, document, as_one_hot=False
                )
                sample_z = torch.Tensor(sample_z).to(prob_z.device).float()
        else:
            sample_z = sampler.sample()

        sample_z = sample_z * mask
        reduced_document = self.select_tokens(document, sample_z)

        encoder_dict = self._encoder(document=reduced_document, query=query, label=label, metadata=metadata)

        loss = 0.0 if not self._supervise_rationale else generator_dict["loss"]

        if label is not None:
            assert "loss" in encoder_dict
            loss_sample = F.cross_entropy(encoder_dict["logits"], label, reduction="none")  # (B,)

            log_prob_z = sampler.log_prob(sample_z)  # (B, L)
            log_prob_z_sum = (mask * log_prob_z).sum(-1)  # (B,)

            lasso_loss = util.masked_mean(sample_z, mask == 1, dim=-1)
            censored_lasso_loss = F.relu(lasso_loss - self._desired_length)

            diff = (sample_z[:, 1:] - sample_z[:, :-1]).abs()
            mask_last = mask[:, :-1]
            fused_lasso_loss = diff.sum(-1) / mask_last.sum(-1)

            self._loss_tracks["_lasso_loss"](lasso_loss.mean().item())
            self._loss_tracks["_censored_lasso_loss"](censored_lasso_loss.mean().item())
            self._loss_tracks["_fused_lasso_loss"](fused_lasso_loss.mean().item())
            self._loss_tracks["_base_loss"](loss_sample.mean().item())

            base_loss = loss_sample
            generator_loss = (
                loss_sample.detach()
                + censored_lasso_loss * self._reg_loss_lambda
                + fused_lasso_loss * (self._reg_loss_mu * self._reg_loss_lambda)
            ) * log_prob_z_sum

            self._loss_tracks["_generator_loss"](generator_loss.mean().item())

            loss += (base_loss + generator_loss).mean()

        output_dict["probs"] = encoder_dict["probs"]
        output_dict["predicted_labels"] = encoder_dict["predicted_labels"]

        output_dict["loss"] = loss
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        output_dict["prob_z"] = generator_dict["prob_z"]
        output_dict["predicted_rationale"] = generator_dict["predicted_rationale"]

        self._loss_tracks["_rat_length"](
            util.masked_mean(generator_dict["predicted_rationale"], mask == 1, dim=-1).mean().item()
        )

        self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        new_output_dict["rationale"] = output_dict["rationale"]
        return new_output_dict

    def select_tokens(self, document, sample_z):
        sample_z_cpu = sample_z.cpu().data.numpy()
        assert len(document) == len(sample_z_cpu)
        assert max([len(d["tokens"]) for d in document]) == sample_z_cpu.shape[1]

        new_document = []
        for doc, mask in zip(document, sample_z_cpu):
            mask = mask[: len(doc["tokens"])]
            new_words = [w for w, m in zip(doc["tokens"], mask) if m == 1]

            new_document.append({"tokens": new_words})

        new_document[0]["reader_object"] = document[0]["reader_object"]

        return new_document

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(BernoulliGenEncClassifier, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k, v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset) for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        if self._supervise_rationale:
            loss_metrics.update(self._generator.get_metrics(reset))

        return loss_metrics
