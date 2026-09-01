from pathlib import Path

files = [
    Path('streamlit_app.py'),
    Path('src/predictor.py'),
    *list(Path('src/models').glob('*.py')),
    *list(Path('src/ui').glob('*.py')),
]

print(f"{'File':<36} | {'Lines':<6} | {'Status':<8}")
print("-" * 56)
all_pass = True
for f in sorted(files):
    with open(f, 'r', encoding='utf-8') as fp:
        n_lines = len(fp.readlines())
    status = "OK" if n_lines <= 150 else "EXCEEDS"
    if n_lines > 150:
        all_pass = False
    print(f"{str(f):<36} | {n_lines:<6} | {status:<8}")

print("-" * 56)
print(f"Overall <= 150 Lines Guarantee: {'PASSED' if all_pass else 'FAILED'}")
