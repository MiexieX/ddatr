#!/usr/bin/env python3
"""Parse DDaTR train.txt into a clean per-epoch metric table.

Usage:
    python parse_train_log.py [train.txt]

Handles both the old trainer format (bare "epoch: N" blocks) and the
new format (delimited "========== Epoch N/10 results ==========" blocks).
Prints one line per epoch plus the final best-epoch summary.
"""
import sys
import re

KEY_RE = re.compile(r'^\s*([A-Za-z_0-9]+)\s*:\s*(\S+)\s*$')


def parse(path):
    blocks = []          # list of (order, dict)
    cur = None
    in_summary = False   # inside the trailing "Best results" summary block
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            s = line.strip()
            if s.startswith('===== Best results') or s.startswith('Best results'):
                in_summary = True
                cur = None          # the summary block is not an epoch
                continue
            if in_summary:
                continue
            # new-style delimited block
            if s.startswith('========== Epoch'):
                m = re.search(r'Epoch (\d+)/', s)
                cur = {'epoch': int(m.group(1)) if m else None}
                blocks.append(cur)
                continue
            # old-style block start
            if s.startswith('epoch') and s.split()[0] == 'epoch':
                m = re.search(r'epoch\s*:\s*(\d+)', s)
                cur = {'epoch': int(m.group(1)) if m else None}
                blocks.append(cur)
                continue
            if cur is None:
                continue
            m = KEY_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    cur[key] = float(val)
                except ValueError:
                    cur[key] = val
    return blocks


def fmt(v, w=9):
    if isinstance(v, float):
        return '{:.4f}'.format(v).rjust(w)
    return str(v).rjust(w)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'train.txt'
    blocks = parse(path)
    if not blocks:
        print('no epoch blocks found in', path)
        return 1

    cols = ['epoch', 'train_loss', 'val_ce_f1', 'test_BLEU_1', 'test_BLEU_4',
            'test_METEOR', 'test_ROUGE_L', 'test_ce_f1']
    header = '  '.join(c.rjust(9) for c in cols)
    print(header)
    print('-' * len(header))

    best = None
    for b in blocks:
        if b.get('epoch') is None:
            continue
        row = [b.get(c) for c in cols]
        print('  '.join(fmt(v) for v in row))
        bef = b.get('val_ce_f1')
        if best is None or (bef is not None and best[1] < bef):
            best = (b.get('epoch'), bef)

    if best:
        print('-' * len(header))
        print('BEST by val_ce_f1: epoch {} (val_ce_f1 = {:.4f})'.format(*best))
    return 0


if __name__ == '__main__':
    sys.exit(main())