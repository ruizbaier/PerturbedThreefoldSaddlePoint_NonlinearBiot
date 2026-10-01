'''
Post-processing for bdgrv_parameter_robustness.py: writes pgfplotstable data files and standalone pgfplots
figures (4x4 panels: rows = lambda, columns = kappa0; in each panel colour = alpha, marker = c0),
in the spirit of helm_fluxbcs_Krange_bdm1.tex.

Run standalone (python3 robustness_plot.py [outdir]) to merge partial csv files produced by running
bdgrv_parameter_robustness.py with lambda indices, and to regenerate tables/figures without recomputing.

Metrics: rel_<var> = ||e(var)||/||var|| in the norms of Section 6 (rel_total = sum over var), and
         w_<var>   = ||e(var)||_W/|||x|||_W in the parameter-weighted norms (w_total = |||e|||_W/|||x|||_W).
'''
import os, sys, glob, csv, subprocess
import numpy as np

OUTDIR = 'outputs/robustness'
LMBDA_VALS = [1.0, 1.e3, 1.e6, 1.e9]
SMALL_VALS = [1.0, 1.e-3, 1.e-6, 1.e-9]
NAMES = ['eta', 'xi', 'p', 'phi', 'sig', 'u', 'gam']
METRICS = ['w_total']   # metric(s) for which data tables and figures are produced (all are kept in the csv)

# marker for each value of c0 (drawn from largest to smallest so that overlapping curves remain visible)
MARKS = {1.0: ('*', 3.6), 1.e-3: ('square*', 2.9), 1.e-6: ('triangle*', 2.6), 1.e-9: ('diamond*', 2.0)}

FIELDS = ['lmbda', 'kappa0', 'alpha', 'c0', 'level', 'dofs', 'h', 'ht'] + \
         ['e_'+nm for nm in NAMES] + ['rel_'+nm for nm in NAMES] + ['rel_total'] + \
         ['w_'+nm for nm in NAMES] + ['w_total', 'algres']

def colname(lv, kv, av, cv):
    return 'L%.0e_K%.0e_A%.0e_C%.0e' % (lv, kv, av, cv)

def pow10(x):
    e = int(round(np.log10(x)))
    return '1' if e == 0 else '10^{%d}' % e

def write_long_csv(results, fname):
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r)

def read_long_csv(fname):
    with open(fname) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in r:
            r[key] = int(r[key]) if key in ('level', 'dofs') else float(r[key])
    return rows

def write_tables_and_figures(results, outdir=OUTDIR, compile_pdf=True):
    levels = sorted(set(r['level'] for r in results))
    dofs = [next(r['dofs'] for r in results if r['level'] == lev) for lev in levels]
    lookup = {(r['lmbda'], r['kappa0'], r['alpha'], r['c0'], r['level']): r for r in results}
    combos = sorted(set((r['lmbda'], r['kappa0'], r['alpha'], r['c0']) for r in results))
    for metric in METRICS:
        key = metric
        fname = os.path.join(outdir, 'robustness_%s.txt' % metric)
        with open(fname, 'w') as f:
            f.write('dofs ' + ' '.join(colname(*c) for c in combos) + '\n')
            for lev, nd in zip(levels, dofs):
                f.write('%d ' % nd + ' '.join('%.6e' % lookup[c + (lev,)][key] for c in combos) + '\n')
        vals = np.array([r[key] for r in results])
        # O(h) reference line (h ~ DOF^{-1/2} in 2D), half a decade below the smallest error of each panel on the
        # finest mesh
        refs = {}
        for lv in LMBDA_VALS:
            for kv in SMALL_VALS:
                yfin = min(lookup[c + (levels[-1],)][key] for c in combos if c[0] == lv and c[1] == kv)
                y1 = yfin/10**0.5
                refs[(lv, kv)] = (dofs[0], y1*(dofs[-1]/dofs[0])**0.5, dofs[-1], y1)
        write_figure(metric, os.path.basename(fname), vals[vals > 0].min(), vals.max(), outdir, refs)
        if compile_pdf:
            subprocess.run(['pdflatex', '-interaction=nonstopmode', 'robustness_%s.tex' % metric],
                           cwd=outdir, stdout=subprocess.DEVNULL)
            for ext in ('aux', 'log'):
                os.remove(os.path.join(outdir, 'robustness_%s.%s' % (metric, ext)))

# discrete rainbow: one colour per value of alpha, from alpha = 1e-9 (blue) to alpha = 1 (red). Curves get explicit
# colours and the colorbar is drawn with solid rectangles (no PDF shadings, which some viewers interpolate)
RAINBOW4 = [(31, 78, 204), (38, 166, 72), (245, 150, 20), (214, 39, 40)]

def write_figure(metric, datafile, vmin, vmax, outdir=OUTDIR, refs=None):
    '''refs: dict (lambda, kappa0) -> (x0, y0, x1, y1), end points of the O(h) reference line in each panel'''
    if refs:
        vmin = min(vmin, min(min(r[1], r[3]) for r in refs.values()))
    ymin = 10**np.floor(np.log10(vmin)); ymax = 10**np.ceil(np.log10(vmax))
    metas = [int(round(np.log10(av))) for av in SMALL_VALS]    # -> 0, -3, -6, -9
    step = abs(metas[0] - metas[1])
    colordefs = '\n'.join(r'\definecolor{alpha%d}{RGB}{%d,%d,%d}' % ((-m,) + RAINBOW4[b]) for b, m in enumerate(sorted(metas)))
    L = []
    L.append(r'''\documentclass{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.16}
\newcommand{\re}{\mathrm{e}}
\usetikzlibrary{calc}
%s
\begin{document}
\begin{tikzpicture}
\pgfplotstableread{%s}{\data}
\begin{groupplot}[group style={group size=4 by 4, horizontal sep=4pt, vertical sep=4pt,
      xticklabels at=edge bottom, yticklabels at=edge left},
    width=3.8cm, height=3.3cm, xmode=log, ymode=log, ymin=%g, ymax=%g,
    tickpos=left, ytick align=inside, xtick align=inside,
    tick label style={font=\small}, label style={font=\normalsize}, title style={font=\normalsize},
    every axis plot/.append style={line width=0.6pt},
  ]''' % (colordefs, datafile, ymin, ymax))
    for i, lv in enumerate(LMBDA_VALS):
        for j, kv in enumerate(SMALL_VALS):
            opts = []
            if i == 0:
                opts.append(r'title={$\kappa_0=%s$}' % pow10(kv))
            if j == 0:
                opts.append(r'ylabel={$\lambda=%s$}' % pow10(lv))
            if i == len(LMBDA_VALS)-1:
                opts.append(r'xlabel={\texttt{DOF}}')
            if i == 0 and j == len(SMALL_VALS)-1:
                opts.append(r'legend style={at={(1.08,1.0)}, anchor=north west, font=\small}, '
                            r'legend cell align=left')
            L.append(r'\nextgroupplot[%s]' % ', '.join(opts))
            if refs:
                x0, y0, x1, y1 = refs[(lv, kv)]
                L.append(r'\addplot[black, dashed, thick, forget plot] coordinates {(%g,%g) (%g,%g)};' % (x0, y0, x1, y1))
            for cv in SMALL_VALS:
                mk, ms = MARKS[cv]
                for av in SMALL_VALS:
                    col = colname(lv, kv, av, cv)
                    meta = int(round(np.log10(av)))
                    L.append(r'\addplot[color=alpha%d, mark=%s, mark size=%.1f, forget plot] table[x=dofs, y=%s] {\data};'
                             % (-meta, mk, 0.85*ms, col))
            if i == 0 and j == len(SMALL_VALS)-1:
                for cv in SMALL_VALS:
                    mk, ms = MARKS[cv]
                    L.append(r'\addlegendimage{only marks, mark=%s, mark size=%.1f, gray}' % (mk, 0.85*ms))
                    L.append(r'\addlegendentry{$c_0=%s$}' % pow10(cv))
                if refs:
                    L.append(r'\addlegendimage{black, dashed, thick}')
                    L.append(r'\addlegendentry{$\mathcal{O}(h)$}')
    L.append(r'\end{groupplot}')
    # colorbar: solid blocks to the right of rows 3-4 of the last column
    nb = len(SMALL_VALS)
    L.append(r'\coordinate (cbtop) at ($(group c4r3.north east)+(0.45cm,0)$);')
    L.append(r'\coordinate (cbbot) at ($(group c4r4.south east)+(0.45cm,0)$);')
    for b, m in enumerate(sorted(metas)):   # bottom block = smallest alpha
        L.append(r'\fill[alpha%d] ($(cbbot)!%g!(cbtop)$) rectangle ($(cbbot)!%g!(cbtop)+(0.35cm,0)$);' % (-m, b/nb, (b+1)/nb))
        L.append(r'\node[anchor=west, font=\small] at ($(cbbot)!%g!(cbtop)+(0.35cm,0)$) {$%d$};' % ((b+0.5)/nb, m))
    L.append(r'\draw ($(cbbot)$) rectangle ($(cbtop)+(0.35cm,0)$);')
    L.append(r'\node[rotate=90, anchor=north] at ($(cbbot)!0.5!(cbtop)+(1.35cm,0)$) {$\log_{10}\alpha$};')
    L.append(r'''\end{tikzpicture}
\end{document}''')
    with open(os.path.join(outdir, 'robustness_%s.tex' % metric), 'w') as f:
        f.write('\n'.join(L) + '\n')

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else OUTDIR
    parts = sorted(glob.glob(os.path.join(outdir, 'robustness_long_part*.csv')))
    if parts:
        results = sum((read_long_csv(p) for p in parts), [])
        write_long_csv(results, os.path.join(outdir, 'robustness_long.csv'))
    else:
        results = read_long_csv(os.path.join(outdir, 'robustness_long.csv'))
    write_tables_and_figures(results, outdir)
