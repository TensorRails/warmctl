"""Per-stage snapshot and restore timings from lane.sh logs.
Usage: timings.py DIAG_DIR [OUT.json]"""
import json, os, re, sys

STAMP = re.compile(r'^\[\s*([0-9.]+)s\] (.*)$')
SNAP = [('engine ready', r'engine (ready|healthy|up)'), ('warmup done', r'warmup \d+/(\d+) .*'),
        ('canary', r'canary recorded'), ('export', r'weights exported'), ('sleep', r'engine asleep'),
        ('cuda', r'CUDA checkpointed'), ('criu dump', r'CRIU dump'), ('trees', r'trees archived'),
        ('published', r'(published|snapshot written|done)')]
REST = [('started', r'receiver .* started'), ('criu', r'CRIU restore'), ('cuda', r'CUDA restored'),
        ('reload', r'weights reloaded'), ('canary', r'canary'), ('serving', r'^serving ')]


def stages(path, marks):
    out, last = {}, {}
    for line in open(path, errors='replace'):
        m = STAMP.match(line.strip())
        if not m:
            m2 = re.match(r'^(serving started|first host-gateway token) ([0-9.]+)s', line.strip())
            if m2:
                out[m2.group(1)] = float(m2.group(2))
            continue
        t, text = float(m.group(1)), m.group(2)
        for name, pat in marks:
            if re.search(pat, text):
                last[name] = (t, text)
    out['stages'] = {k: v for k, v in last.items()}
    return out


def main():
    diag = sys.argv[1]
    rows = []
    for lane in sorted(os.listdir(diag)):
        d = os.path.join(diag, lane)
        snap = os.path.join(d, 'snapshot.log')
        if not os.path.exists(snap):
            continue
        row = {'lane': lane, 'snapshot': stages(snap, SNAP), 'restores': {}}
        for f in sorted(os.listdir(d)):
            if re.match(r'.*-r\d+\.log$', f):
                row['restores'][f[:-4]] = stages(os.path.join(d, f), REST)
        rows.append(row)
    for r in rows:
        s = r['snapshot']['stages']
        print('%-6s snapshot: %s' % (r['lane'], ', '.join('%s %.1fs' % (k, v[0]) for k, v in s.items())))
        for rid, x in r['restores'].items():
            st = x['stages']
            print('       %-9s %s | serving %s, first token %s' % (
                rid, ', '.join('%s %.1fs' % (k, v[0]) for k, v in st.items()),
                x.get('serving started'), x.get('first host-gateway token')))
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], 'w'), indent=1)


main()
