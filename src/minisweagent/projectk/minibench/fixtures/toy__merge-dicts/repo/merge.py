def merge_counts(a, b):
    out = {}
    for k, v in a.items():
        out[k] = v + b.get(k, 0)
    return out
