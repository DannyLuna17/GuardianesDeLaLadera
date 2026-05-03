from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rmse(predictions: list[float], targets: list[float]) -> float:
    if not predictions:
        return 0.0
    squared_error = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets, strict=True)
    )
    return (squared_error / len(predictions)) ** 0.5


def _sse(sum_values: float, sum_squared_values: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return max(sum_squared_values - ((sum_values**2) / count), 0.0)


def _fit_regression_tree(
    *,
    feature_names: list[str],
    feature_matrix: list[list[float]],
    targets: list[float],
    sample_indices: list[int],
    max_depth: int,
    min_leaf_size: int,
    min_split_gain: float,
    depth: int = 0,
) -> dict[str, Any]:
    node_targets = [targets[index] for index in sample_indices]
    node_value = _mean(node_targets)
    leaf = {
        "node_type": "leaf",
        "value": round(node_value, 6),
        "samples": len(sample_indices),
        "depth": depth,
    }
    if depth >= max_depth or len(sample_indices) < (min_leaf_size * 2):
        return leaf

    node_sum = sum(node_targets)
    node_sq_sum = sum(target * target for target in node_targets)
    current_loss = _sse(node_sum, node_sq_sum, len(sample_indices))

    best_split: dict[str, Any] | None = None
    for feature_index, feature_name in enumerate(feature_names):
        ordered = sorted(
            (
                feature_matrix[index][feature_index],
                targets[index],
                index,
            )
            for index in sample_indices
        )
        if len(ordered) < (min_leaf_size * 2):
            continue

        prefix_sum: list[float] = []
        prefix_sq_sum: list[float] = []
        running_sum = 0.0
        running_sq_sum = 0.0
        for _, target, _ in ordered:
            running_sum += target
            running_sq_sum += target * target
            prefix_sum.append(running_sum)
            prefix_sq_sum.append(running_sq_sum)

        total_sum = prefix_sum[-1]
        total_sq_sum = prefix_sq_sum[-1]
        sample_count = len(ordered)

        for split_position in range(min_leaf_size, sample_count - min_leaf_size + 1):
            if split_position >= sample_count:
                break
            left_value = ordered[split_position - 1][0]
            right_value = ordered[split_position][0]
            if left_value == right_value:
                continue

            left_count = split_position
            right_count = sample_count - split_position
            left_sum = prefix_sum[split_position - 1]
            left_sq_sum = prefix_sq_sum[split_position - 1]
            right_sum = total_sum - left_sum
            right_sq_sum = total_sq_sum - left_sq_sum

            gain = current_loss - (
                _sse(left_sum, left_sq_sum, left_count)
                + _sse(right_sum, right_sq_sum, right_count)
            )
            if gain <= min_split_gain:
                continue

            threshold = (left_value + right_value) / 2
            if best_split is None or gain > best_split["gain"]:
                best_split = {
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                    "ordered": ordered,
                    "split_position": split_position,
                }

    if best_split is None:
        return leaf

    ordered = best_split["ordered"]
    split_position = int(best_split["split_position"])
    left_indices = [item[2] for item in ordered[:split_position]]
    right_indices = [item[2] for item in ordered[split_position:]]
    if not left_indices or not right_indices:
        return leaf

    return {
        "node_type": "split",
        "feature": str(best_split["feature"]),
        "feature_index": int(best_split["feature_index"]),
        "threshold": round(float(best_split["threshold"]), 6),
        "gain": round(float(best_split["gain"]), 6),
        "value": round(node_value, 6),
        "samples": len(sample_indices),
        "depth": depth,
        "left": _fit_regression_tree(
            feature_names=feature_names,
            feature_matrix=feature_matrix,
            targets=targets,
            sample_indices=left_indices,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            min_split_gain=min_split_gain,
            depth=depth + 1,
        ),
        "right": _fit_regression_tree(
            feature_names=feature_names,
            feature_matrix=feature_matrix,
            targets=targets,
            sample_indices=right_indices,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            min_split_gain=min_split_gain,
            depth=depth + 1,
        ),
    }


def predict_regression_tree(
    tree: dict[str, Any],
    feature_row: dict[str, float] | list[float],
    *,
    with_path: bool = False,
) -> float | tuple[float, list[dict[str, Any]]]:
    path: list[dict[str, Any]] = []
    node = tree
    while node.get("node_type") == "split":
        feature_name = str(node["feature"])
        if isinstance(feature_row, dict):
            feature_value = float(feature_row.get(feature_name, 0.0))
        else:
            feature_value = float(feature_row[int(node["feature_index"])])
        branch = "left" if feature_value <= float(node["threshold"]) else "right"
        if with_path:
            path.append(
                {
                    "feature": feature_name,
                    "threshold": float(node["threshold"]),
                    "gain": float(node.get("gain", 0.0)),
                    "branch": branch,
                    "feature_value": feature_value,
                }
            )
        node = node[branch]

    value = float(node.get("value", 0.0))
    if with_path:
        return value, path
    return value


def predict_gradient_boosted_ensemble(
    ensemble: dict[str, Any],
    feature_row: dict[str, float] | list[float],
    *,
    with_contributions: bool = False,
) -> float | tuple[float, dict[str, float]]:
    base_score = float(ensemble.get("base_score", 0.0))
    learning_rate = float(ensemble.get("learning_rate", 0.1))
    prediction = base_score
    contributions = {
        feature_name: 0.0 for feature_name in ensemble.get("feature_order") or []
    }
    for tree in ensemble.get("trees") or []:
        if with_contributions:
            leaf_value, path = predict_regression_tree(tree, feature_row, with_path=True)
            tree_contribution = learning_rate * float(leaf_value)
            prediction += tree_contribution
            if path:
                path_gain_total = sum(
                    max(float(item.get("gain", 0.0)), 0.0) for item in path
                )
                for item in path:
                    if path_gain_total > 1e-12:
                        weight = max(float(item.get("gain", 0.0)), 0.0) / path_gain_total
                    else:
                        weight = 1 / len(path)
                    feature_name = str(item["feature"])
                    contributions[feature_name] = contributions.get(feature_name, 0.0) + (
                        tree_contribution * weight
                    )
            continue

        prediction += learning_rate * float(
            predict_regression_tree(tree, feature_row, with_path=False)
        )
    if with_contributions:
        return prediction, {key: round(value, 6) for key, value in contributions.items()}
    return prediction


def feature_gain_importance(ensemble: dict[str, Any]) -> dict[str, float]:
    importance = {
        feature_name: 0.0 for feature_name in ensemble.get("feature_order") or []
    }
    for tree in ensemble.get("trees") or []:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.get("node_type") != "split":
                continue
            feature_name = str(node["feature"])
            importance[feature_name] = importance.get(feature_name, 0.0) + float(
                node.get("gain", 0.0)
            )
            stack.append(node["left"])
            stack.append(node["right"])
    return {key: round(value, 6) for key, value in importance.items()}


def train_gradient_boosted_regressor(
    *,
    feature_names: list[str],
    train_matrix: list[list[float]],
    train_targets: list[float],
    validation_matrix: list[list[float]] | None = None,
    validation_targets: list[float] | None = None,
    learning_rate: float = 0.1,
    estimator_count: int = 24,
    max_depth: int = 3,
    min_leaf_size: int = 2,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 4,
) -> dict[str, Any]:
    if not train_matrix or not train_targets:
        raise ValueError("Gradient boosting requires non-empty training data.")

    base_score = _mean(train_targets)
    train_predictions = [base_score for _ in train_targets]
    validation_predictions = (
        [_mean(train_targets) for _ in (validation_targets or [])]
        if validation_targets is not None
        else None
    )
    trees: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    best_tree_count = 0
    best_validation_rmse = float("inf")
    rounds_without_improvement = 0

    for iteration in range(1, estimator_count + 1):
        residuals = [
            target - prediction
            for target, prediction in zip(train_targets, train_predictions, strict=True)
        ]
        tree = _fit_regression_tree(
            feature_names=feature_names,
            feature_matrix=train_matrix,
            targets=residuals,
            sample_indices=list(range(len(train_matrix))),
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            min_split_gain=min_split_gain,
        )
        trees.append(tree)

        train_updates = [
            float(predict_regression_tree(tree, feature_row))
            for feature_row in train_matrix
        ]
        train_predictions = [
            prediction + (learning_rate * update)
            for prediction, update in zip(train_predictions, train_updates, strict=True)
        ]
        validation_rmse = None
        if validation_matrix is not None and validation_predictions is not None:
            validation_updates = [
                float(predict_regression_tree(tree, feature_row))
                for feature_row in validation_matrix
            ]
            validation_predictions = [
                prediction + (learning_rate * update)
                for prediction, update in zip(
                    validation_predictions, validation_updates, strict=True
                )
            ]
            validation_rmse = round(
                _rmse(validation_predictions, validation_targets or []), 6
            )
            if validation_rmse < (best_validation_rmse - 1e-9):
                best_validation_rmse = validation_rmse
                best_tree_count = len(trees)
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
        else:
            best_tree_count = len(trees)

        history.append(
            {
                "iteration": iteration,
                "train_rmse": round(_rmse(train_predictions, train_targets), 6),
                "validation_rmse": validation_rmse,
                "tree_depth": _tree_depth(tree),
                "leaf_count": _leaf_count(tree),
            }
        )
        if (
            validation_matrix is not None
            and early_stopping_rounds > 0
            and rounds_without_improvement >= early_stopping_rounds
        ):
            break

    effective_tree_count = best_tree_count or len(trees)
    final_trees = trees[:effective_tree_count]
    ensemble = {
        "base_score": round(base_score, 6),
        "learning_rate": round(learning_rate, 6),
        "feature_order": feature_names,
        "trees": final_trees,
    }
    return {
        "base_score": round(base_score, 6),
        "learning_rate": round(learning_rate, 6),
        "requested_tree_count": estimator_count,
        "effective_tree_count": effective_tree_count,
        "trees": final_trees,
        "feature_importance": feature_gain_importance(ensemble),
        "training_history": history,
        "early_stopping_triggered": effective_tree_count < estimator_count,
        "best_validation_rmse": (
            round(best_validation_rmse, 6)
            if validation_matrix is not None and best_validation_rmse < float("inf")
            else None
        ),
    }


def _tree_depth(tree: dict[str, Any]) -> int:
    if tree.get("node_type") != "split":
        return int(tree.get("depth", 0))
    return max(_tree_depth(tree["left"]), _tree_depth(tree["right"]))


def _leaf_count(tree: dict[str, Any]) -> int:
    if tree.get("node_type") != "split":
        return 1
    return _leaf_count(tree["left"]) + _leaf_count(tree["right"])
