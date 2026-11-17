"""Horizontal-strip averaged strain profiles: atomic-coordinate CSV vs GPA.

Dependencies: numpy scipy matplotlib (Pillow only for optional STEM intensity).
Edit the configuration below and run: python strain_profiles.py
Paths are relative to this script unless absolute. Source files are read only.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------- USER CONFIGURATION --------------------------------------
BASE = Path(r"E:\Articol MAM\Prelucrare Strain ++").resolve().parent
CSV_PATH = BASE / 'Prelucrare Strain ++' / 'superlattice_strain.csv'
GPA_XX_PATH = BASE / 'Prelucrare Strain ++' / 'epsxx.bin'
GPA_YY_PATH = BASE / 'Prelucrare Strain ++' / 'epsyy.bin'
GPA_SHAPE = (681, 790)                 # (height, width), NOT (width, height)
GPA_DTYPE = '<f8'                      # little-endian float64, no header
GPA_MULTIPLIER = 1.0                   # 1 for fractional strain; .01 if stored %
CSV_MULTIPLIER = 1.0                   # 1 for fractional strain; .01 if stored %
X_RANGE = (300, 600)                    # horizontal strip [x0, x1), crop pixels
Y_RANGE = (50, 630)                    # depth range [y0, y1), crop pixels
BIN_HEIGHT_PX = 10                    # vertical averaging; 1 = one profile point/row
MIN_VALID_FRACTION = 0.8              # discard bins with inadequate common support
MAX_INTERPOLATION_DISTANCE_PX = 12.0  # do not bridge large unsupported holes
PIXEL_SIZE_NM = None                  # enter ONLY after confirming crop calibration
Y_ORIGIN_PX = 0.0                     # plotted depth = (y - origin) * pixel size
STEM_CROP_PATH = None                 # optional scalar TIFF of EXACT same crop
# Optional per-layer metrics (set actual boundaries, not illustrative guesses):
# LAYER_RANGES_PX = [('Layer 1', 550, 630), ('Layer 2', 370, 520), ...]
LAYER_RANGES_PX = []
OUT = BASE / 'strain_profiles_output'
# --------------------------------------------------------------------------


def write_records(path, records):
    if not records:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def agreement(p, g):
    """Inputs in percent; all differences in percentage points (pp)."""
    p, g = np.asarray(p), np.asarray(g)
    ok = np.isfinite(p) & np.isfinite(g)
    p, g = p[ok], g[ok]
    if not len(p):
        raise ValueError('No common finite values')
    d = p-g
    r = float(np.corrcoef(p, g)[0, 1]) if len(p)>1 and np.std(p)>0 and np.std(g)>0 else None
    return dict(n=int(len(p)), program_mean_percent=float(p.mean()),
                gpa_mean_percent=float(g.mean()), bias_pp=float(d.mean()),
                mae_pp=float(np.abs(d).mean()), rmse_pp=float(np.sqrt(np.mean(d*d))),
                difference_sd_pp=float(np.std(d)), pearson_r=r)


def load_csv(path):
    with path.open(newline='') as f:
        rows = list(csv.reader(f))
    header = [s.strip().lstrip('\ufeff') for s in rows[0]]
    indexes = [header.index(k) for k in ('x_px', 'y_px', 'exx', 'eyy')]
    vals = []
    for row in rows[1:]:
        if not row:
            continue
        # The supplied file has an unused trailing header without row values.
        if len(row) <= max(indexes):
            raise ValueError('CSV row is missing required position/strain fields')
        vals.append([float(row[i]) for i in indexes])
    return np.asarray(vals)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h, w = GPA_SHAPE
    x0, x1 = X_RANGE
    y0, y1 = Y_RANGE
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI outside GPA map')
    if BIN_HEIGHT_PX < 1 or int(BIN_HEIGHT_PX) != BIN_HEIGHT_PX:
        raise ValueError('BIN_HEIGHT_PX must be a positive integer')
    if PIXEL_SIZE_NM is not None and PIXEL_SIZE_NM <= 0:
        raise ValueError('PIXEL_SIZE_NM must be positive')
    a = load_csv(CSV_PATH)
    pos, eps = a[:, :2], a[:, 2:]*CSV_MULTIPLIER
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(eps).all(axis=1)
    pos, eps = pos[finite], eps[finite]
    if len(pos) < 3:
        raise ValueError('Not enough valid atomic sites')
    gy, gx = np.mgrid[y0:y1, x0:x1]
    points = np.column_stack((gx.ravel(), gy.ravel()))
    # One piecewise-linear interpolation only, with NO Gaussian smoothing.
    interp = LinearNDInterpolator(pos, eps, fill_value=np.nan)
    interp_eps = interp(points).reshape(gy.shape+(2,))
    distance = cKDTree(pos).query(points)[0].reshape(gy.shape)
    support = np.isfinite(interp_eps).all(axis=2) & (distance <= MAX_INTERPOLATION_DISTANCE_PX)
    maps = {}
    for i, (c, path) in enumerate([('xx', GPA_XX_PATH), ('yy', GPA_YY_PATH)]):
        arr = np.fromfile(path, dtype=GPA_DTYPE)
        if arr.size != h*w:
            raise ValueError(f'{path.name}: expected {h*w} values, got {arr.size}')
        gpa = arr.reshape(GPA_SHAPE)*GPA_MULTIPLIER
        prog = interp_eps[:, :, i]
        g = gpa[y0:y1, x0:x1]
        mask = support & np.isfinite(g)
        maps[c] = dict(program=prog, gpa=g, mask=mask, full_gpa=gpa)

    profiles, all_metrics, layer_metrics = [], [], []
    for c in ('yy', 'xx'):
        v = maps[c]
        for lo in range(y0, y1, BIN_HEIGHT_PX):
            hi = min(lo+BIN_HEIGHT_PX, y1)
            sl = slice(lo-y0, hi-y0)
            m = v['mask'][sl]
            coverage = float(m.mean())
            if coverage < MIN_VALID_FRACTION or not m.any():
                continue
            p, g = v['program'][sl][m]*100, v['gpa'][sl][m]*100
            center = (lo+hi-1)/2
            plot_y = center if PIXEL_SIZE_NM is None else (center-Y_ORIGIN_PX)*PIXEL_SIZE_NM
            profiles.append(dict(component=c, y_start_px=lo, y_end_px=hi,
                                 y_center_px=center, depth=plot_y, n_pixels=int(m.sum()),
                                 common_fraction=coverage, program_percent=float(p.mean()),
                                 gpa_percent=float(g.mean()), delta_pp=float((p-g).mean()),
                                 program_spatial_sd_percent=float(p.std()),
                                 gpa_spatial_sd_percent=float(g.std())))
        rows = [r for r in profiles if r['component']==c]
        if not rows:
            raise ValueError('No profile bins passed support threshold')
        all_metrics.append(dict(component=c, population='profile_bin_means',
                                **agreement([r['program_percent'] for r in rows], [r['gpa_percent'] for r in rows])))
        # Raw paired-site values are reported separately from profile metrics.
        at_sites = map_coordinates(v['full_gpa'], [pos[:,1],pos[:,0]], order=1, mode='constant', cval=np.nan)
        sm=(pos[:,0]>=x0)&(pos[:,0]<x1)&(pos[:,1]>=y0)&(pos[:,1]<y1)&np.isfinite(at_sites)
        all_metrics.append(dict(component=c,population='individual_atomic_sites',
                                **agreement(eps[sm, 0 if c=='xx' else 1]*100,at_sites[sm]*100)))
        for label, lo, hi in LAYER_RANGES_PX:
            if not (y0<=lo<hi<=y1):
                raise ValueError(f'Layer {label}: interval must be inside Y_RANGE')
            m=v['mask']&(gy>=lo)&(gy<hi)
            layer_metrics.append(dict(layer=label, component=c,y_start_px=lo,y_end_px=hi,
                                      **agreement(v['program'][m]*100,v['gpa'][m]*100)))
    write_records(OUT/'profiles.csv',profiles)
    write_records(OUT/'metrics.csv',all_metrics)
    write_records(OUT/'layer_metrics.csv',layer_metrics)

    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(12.8,8),sharex='col')
    fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.88,wspace=.25,hspace=.30)
    colors = ('#1679b0','#ed7a14')
    xlabel = 'y in crop (px; increasing downward)' if PIXEL_SIZE_NM is None else 'Depth (nm; increasing downward)'
    for j,c in enumerate(('yy','xx')):
        rows=[r for r in profiles if r['component']==c]
        yy=np.array([r['depth'] for r in rows])
        p=np.array([r['program_percent'] for r in rows]);g=np.array([r['gpa_percent'] for r in rows])
        ax=axes[0,j]
        ax.vlines(yy,np.minimum(p,g),np.maximum(p,g),color='0.72',lw=.9,zorder=1)
        ax.scatter(yy,p,s=34,facecolors='none',edgecolors=colors[0],lw=1.5,label='Present workflow',zorder=3)
        ax.scatter(yy,g,s=34,marker='x',color=colors[1],lw=1.4,label='GPA (Strain++)',zorder=3)
        ax.axhline(0,color='0.8',lw=.7)
        ax.set(title=rf'$\varepsilon_{{{c}}}$: means over the same horizontal strip',ylabel='Strain (%)')
        ax.legend(frameon=False,fontsize=9)
        ax=axes[1,j]
        ax.plot(yy,p-g,'o-',ms=3.5,color='#593c85',lw=1)
        ax.axhline(0,color='0.3',ls='--',lw=1)
        q=next(q for q in all_metrics if q['component']==c and q['population']=='profile_bin_means')
        ax.set(xlabel=xlabel,ylabel='Workflow − GPA (percentage points)',title=f"Profile bias = {q['bias_pp']:+.3f} pp; RMSE = {q['rmse_pp']:.3f} pp")
        ax.grid(alpha=.15)
    fig.suptitle(f'Horizontal average: x = {x0}:{x1} px; depth bins = {BIN_HEIGHT_PX} px',fontsize=15)
    fig.text(.5,.012,'Piecewise-linear interpolation of atomic values. Identical pixel support. Original reference levels retained.',ha='center',fontsize=9,color='0.3')
    fig.savefig(OUT/'paired_profiles.png',dpi=220)
    fig.savefig(OUT/'paired_profiles.svg')
    plt.close(fig)

    # Display coordinates to document the actual strip used, without guessing
    # the relationship to the 1024x1024 source TIFF.
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    for ax,c in zip(axes,('yy','xx')):
        v=maps[c]
        ax.imshow(v['full_gpa']*100,cmap='RdBu_r',vmin=-7 if c=='yy' else -2,vmax=7 if c=='yy' else 2,origin='upper')
        ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='black',lw=1.8))
        ax.set(title=rf'GPA $\varepsilon_{{{c}}}$: black box = averaging strip',xlabel='x in crop (px)',ylabel='y in crop (px)')
    fig.savefig(OUT/'strip_selection.png',dpi=180)
    plt.close(fig)

    if STEM_CROP_PATH is not None:
        from PIL import Image
        im=np.asarray(Image.open(STEM_CROP_PATH))
        if im.ndim!=2 or im.shape!=GPA_SHAPE:
            raise ValueError('STEM_CROP_PATH must be a scalar image of the exact GPA crop; crop/align the original first')
        intensity=np.mean(im[y0:y1,x0:x1],axis=1)
        write_records(OUT/'intensity_profile.csv',[dict(y_px=int(k),intensity=float(val)) for k,val in zip(range(y0,y1),intensity)])
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        yy=np.arange(y0,y1)
        if PIXEL_SIZE_NM is not None: yy=(yy-Y_ORIGIN_PX)*PIXEL_SIZE_NM
        ax.plot(yy,intensity,color='0.25');ax.set(xlabel=xlabel,ylabel='Mean image intensity (a.u.)')
        fig.savefig(OUT/'intensity_profile.png',dpi=180);plt.close(fig)

    config=dict(gpa_shape=GPA_SHAPE,gpa_dtype=GPA_DTYPE,gpa_multiplier=GPA_MULTIPLIER,
                csv_multiplier=CSV_MULTIPLIER,x_range=X_RANGE,y_range=Y_RANGE,bin_height_px=BIN_HEIGHT_PX,
                pixel_size_nm=PIXEL_SIZE_NM,min_valid_fraction=MIN_VALID_FRACTION,
                max_interpolation_distance_px=MAX_INTERPOLATION_DISTANCE_PX,layers=LAYER_RANGES_PX,
                valid_atomic_sites=int(finite.sum()))
    (OUT/'configuration.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(all_metrics,indent=2))
    print(f'Outputs: {OUT}')


if __name__=='__main__':
    main()
