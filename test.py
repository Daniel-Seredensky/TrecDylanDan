
old_reserve = 10
reserve = 15
foo = "additional" if old_reserve is not None else ""
print(f"Requesting {foo} {reserve} tokens")