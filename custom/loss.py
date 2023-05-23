import torch
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import wandb
except ImportError:
    wandb = None

from open_clip.loss import ClipLoss
import ot


def create_loss(args):
    return SemiSupervisedClipLoss(
        args.method,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


class SemiSupervisedClipLoss(ClipLoss):
    def __init__(
            self,
            method,
            pseudo_label_type="ot-image",
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        assert method in ["base", "ours"]
        self.method = method
        self.pseudo_label_type = pseudo_label_type

    def forward(self, image_features, text_features, logit_scale, output_dict=False,
                query_features=None, keyword_features=None, keyword_labels=None):
        device = image_features.device
        losses = dict()  # dict of losses

        # gather tensors over worlds
        if self.world_size > 1:
            dist_kwargs = {
                "local_loss": self.local_loss,
                "gather_with_grad": self.gather_with_grad,
                "rank": self.rank,
                "world_size": self.world_size,
                "use_horovod": self.use_horovod,
            }
            image_features = gather_features(image_features, **dist_kwargs)
            text_features = gather_features(text_features, **dist_kwargs)
            query_features = gather_features(query_features, **dist_kwargs)
            keyword_labels = gather_features(keyword_labels, **dist_kwargs)

        # compute loss
        if self.method == "base":
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T

            labels = self.get_ground_truth(device, image_features.shape[0])
            losses["contrastive_loss"] = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_query = logit_scale * query_features @ text_features.T
            logits_per_text = torch.cat([logits_per_image, logits_per_query]).T

            # supervised CLIP loss
            labels = self.get_ground_truth(device, image_features.shape[0])
            losses["contrastive_loss"] = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

            # caption-level loss
            plan = get_assignments(
                query_features, image_features, text_features, logit_scale, self.pseudo_label_type)
            pseudo_labels = plan @ F.one_hot(labels).float()

            losses["caption_loss"] = (
                soft_cross_entropy(logits_per_query, pseudo_labels)
            ) / 2

            # keyword-level loss
            selected = []
            pseudo_labels_keyword = torch.zeros(len(query_features), len(keyword_features), device=device)
            for query_id, q in enumerate(query_features):
                sample_id = int(plan[query_id].max(dim=0)[1])  # nearest one
                candidates = keyword_labels[sample_id, :, 0].nonzero().flatten().tolist()

                if len(candidates) > 0:
                    selected.append(query_id)
                    if len(candidates) == 1:
                        pseudo_labels_keyword[query_id, candidates[0]] = 1
                    else:
                        k = torch.stack([keyword_features[i] for i in candidates])
                        sim = (q @ k.T * logit_scale).detach()
                        prob = sim / sim.sum()
                        for i in range(len(sim)):
                            pseudo_labels_keyword[query_id, candidates[i]] = prob[i]

            logits_per_query_keyword = logit_scale * query_features @ keyword_features.T
            losses["keyword_loss"] = (
                soft_cross_entropy(logits_per_query_keyword, pseudo_labels_keyword)
            ) / 2

        return losses if output_dict else sum(losses.items())


def get_assignments(query, image, text, logit_scale, pseudo_label_type):
    if pseudo_label_type == "hard-image":
        plan = hard_nn(query, image)
    elif pseudo_label_type == "hard-text":
        plan = hard_nn(query, image)
    elif pseudo_label_type == "soft-image":
        plan = soft_nn(query, image, logit_scale)
    elif pseudo_label_type == "soft-text":
        plan = soft_nn(query, text, logit_scale)
    elif pseudo_label_type == "ot-image":
        plan = ot_plan(query, image, logit_scale)
    elif pseudo_label_type == "ot-text":
        plan = ot_plan(query, image, logit_scale)
    else:
        raise NotImplementedError
    return plan

def hard_nn(query, support):
    _, idx = (query @ support.T).max(dim=1)
    plan = F.one_hot(idx, len(support)).float()
    return plan


def soft_nn(query, support, logit_scale):
    plan = F.softmax(query @ support.T * logit_scale, dim=1)
    return plan


def ot_plan(query, support, logit_scale):
    C = 1 - query @ support.T  # (query, batch)
    reg = 1 / logit_scale  # learned temperature

    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q
    P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=10)

    plan = P / P.sum(dim=1, keepdim=True)
    plan = plan.type_as(support)
    return plan


def soft_cross_entropy(outputs, targets, weight=1.):
    loss = -targets * F.log_softmax(outputs, dim=1)
    return (loss * weight).sum(dim=1).mean()


def gather_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if features is None:
        return features

    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features
