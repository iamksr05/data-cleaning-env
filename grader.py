def compute_score(dataset):
    total = len(dataset)

    if total == 0:
        return 0.0

    # Completeness (no missing)
    missing = sum(1 for row in dataset if None in row.values())
    completeness = 1 - (missing / total)

    # Uniqueness (no duplicates)
    unique_rows = [dict(t) for t in {tuple(row.items()) for row in dataset}]
    uniqueness = len(unique_rows) / total

    # Consistency (lowercase cities)
    consistent = sum(1 for row in dataset if row["city"].islower())
    consistency = consistent / total

    score = (
        0.4 * completeness +
        0.3 * uniqueness +
        0.3 * consistency
    )

    return round(score, 3)