def chunks(items, n):
    out = []
    for i in range(0, len(items), n):
        out.append(items[i : i + n])
    return out
