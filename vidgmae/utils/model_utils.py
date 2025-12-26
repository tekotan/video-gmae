import numpy as np
import torch


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


def compute_tapvid_metrics(
    query_points,
    gt_occluded,
    gt_tracks,
    pred_occluded,
    pred_tracks,
    query_mode="first",
):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)."""
    assert gt_occluded.shape == pred_occluded.shape

    metrics = {}

    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    if query_mode == "first":
        assert gt_occluded.shape[0] == 1, "Expected batch size 1 gt_occluded"
        for i in range(gt_occluded.shape[1]):
            index = np.where(gt_occluded[0, i] == 0)[0]
            index = index[0] if len(index) > 0 else gt_occluded.shape[2]
            evaluation_points[0, i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    metrics["occ_tp"] = np.sum(
        np.equal(pred_occluded, gt_occluded) & gt_occluded & evaluation_points,
        axis=(1, 2),
    )
    metrics["occ_fp"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded))
        & pred_occluded
        & evaluation_points,
        axis=(1, 2),
    )
    metrics["occ_fn"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded))
        & np.logical_not(pred_occluded)
        & evaluation_points,
        axis=(1, 2),
    )

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []

    l2_error = np.sqrt(
        np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        )
    )
    masked_l2_error = l2_error * (1 - gt_occluded)
    avg_distance = np.sum(masked_l2_error) / np.sum(1 - gt_occluded)
    metrics["avg_distance"] = np.array([avg_distance])
    nonzero_masked_error = l2_error[(1 - gt_occluded).astype(bool)]
    assert np.allclose(masked_l2_error.sum(), nonzero_masked_error.sum()), (
        masked_l2_error.sum(),
        nonzero_masked_error.sum(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    assert nonzero_masked_error.size == np.sum(1 - gt_occluded)
    assert np.allclose(avg_distance, nonzero_masked_error.mean()), (
        avg_distance,
        nonzero_masked_error.mean(),
        avg_distance - nonzero_masked_error.mean(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    metrics["median_distance"] = np.array([np.median(nonzero_masked_error)])

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct

        metrics["num_visible"] = count_visible_points
        metrics["num_pts_within_" + str(thresh)] = count_correct

        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics
