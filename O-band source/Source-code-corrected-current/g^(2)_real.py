"""Integrate the measured histogram and plot background-corrected peak areas.

The seven integration windows reproduce the previously used 3-bin areas.
Background is an independently recorded measurement (see g2_config.json),
not fitted to a desired g2 or purity. All plotted values and annotations are
calculated from the data and this background measurement in the same run.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def analyze(data_path, config):
    table = np.loadtxt(data_path, skiprows=1, usecols=(0, 1))
    if table.ndim != 2 or len(table) < 2:
        raise ValueError("The histogram must contain at least two numeric rows.")
    time, counts = table.T
    if not np.all(np.isfinite(table)) or np.any(counts < 0):
        raise ValueError("Histogram coordinates/counts must be finite; counts must be nonnegative.")
    step = np.diff(time)
    if np.any(step <= 0) or not np.allclose(step, step[0]):
        raise ValueError("The histogram must use increasing, equally spaced bin centres.")

    windows = config['integration_windows']
    orders = np.array([w['relative_peak'] for w in windows], dtype=int)
    if len(np.unique(orders)) != len(orders) or np.count_nonzero(orders == 0) != 1:
        raise ValueError("Relative peak indices must be unique and contain exactly one zero.")
    central_index = int(np.flatnonzero(orders == 0)[0])
    background = config['background_measurement']
    n_reference = int(background['reference_window_bins'])
    bg = float(background['counts_per_reference_window'])
    bg_se = float(background['standard_error_per_reference_window'])
    if n_reference <= 0 or not np.isfinite(bg) or not np.isfinite(bg_se) or bg < 0 or bg_se < 0:
        raise ValueError("Invalid measured background or uncertainty.")

    used = np.zeros(len(time), dtype=bool)
    masks, rows = [], []
    for window in windows:
        left, right = window['first_bin_centre'], window['last_bin_centre']
        if not np.any(np.isclose(time, left)) or not np.any(np.isclose(time, right)):
            raise ValueError(f"Integration endpoints {left}, {right} are absent from the data.")
        mask = (time >= left) & (time <= right)
        n_bins = int(mask.sum())
        if n_bins != n_reference:
            raise ValueError("Each integration window must have the measured background's reference width.")
        if np.any(used & mask):
            raise ValueError("Peak integration windows must not overlap.")
        used |= mask
        masks.append(mask)
        # Actual integration: sum the counts in each specified histogram bin.
        raw = float(np.sum(counts[mask]))
        rows.append({
            'relative_peak': int(window['relative_peak']),
            'first_bin_centre': float(left),
            'last_bin_centre': float(right),
            'n_bins': n_bins,
            'bin_counts': ' + '.join(f'{value:g}' for value in counts[mask]),
            'raw_integral': raw,
            'background_subtracted': bg,
            'corrected_integral': raw - bg,
        })

    raw_areas = np.array([row['raw_integral'] for row in rows])
    corrected = raw_areas - bg
    side_mask = orders != 0
    n_side = int(side_mask.sum())
    if n_side < 2:
        raise ValueError("At least two side peaks are required for the side-peak standard error.")
    cc_raw = float(raw_areas[central_index])
    acc_raw = float(np.mean(raw_areas[side_mask]))
    cc = cc_raw - bg
    acc = acc_raw - bg
    if acc <= 0:
        raise ValueError("Mean background-corrected side-peak integral must be positive.")

    normalized = corrected / acc
    g2 = float(normalized[central_index])
    # Central-peak Poisson uncertainty; side-peak SEM from their measured scatter.
    cc_raw_se = float(np.sqrt(cc_raw))
    acc_raw_se = float(np.std(raw_areas[side_mask], ddof=1) / np.sqrt(n_side))
    # g2 = (C - N)/(A - N). N is shared by numerator and denominator:
    # dg/dC = 1/(A-N), dg/dA = -g/(A-N), dg/dN = (g-1)/(A-N).
    # This propagation assumes the recorded N is independent of the peak sums.
    g2_se = float(np.sqrt(cc_raw_se**2 + (g2 * acc_raw_se)**2
                          + ((g2 - 1) * bg_se)**2) / acc)
    purity_percent = 100.0 * (g2 - 1.0)
    purity_se_percent = 100.0 * g2_se
    for row, height in zip(rows, normalized):
        row['normalized_corrected_integral'] = float(height)

    summary = {
        'data_file': str(Path(data_path).resolve()),
        'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'input_coordinate_units': 'as exported; no ns conversion assumed',
        'integration': 'sum of counts at the three bin centres listed for each peak',
        'configuration': config,
        'raw_integrals': raw_areas.tolist(),
        'corrected_integrals': corrected.tolist(),
        'normalized_corrected_integrals': normalized.tolist(),
        'CC_raw': cc_raw, 'CC_raw_standard_error': cc_raw_se,
        'ACC_raw_mean': acc_raw, 'ACC_raw_mean_standard_error': acc_raw_se,
        'CC_corrected': cc, 'ACC_corrected_mean': acc,
        'g2_raw': cc_raw / acc_raw,
        'g2_corrected': g2, 'g2_standard_error': g2_se,
        'purity_percent': purity_percent, 'purity_standard_error_percent': purity_se_percent,
        'annotation': f'Purity:\n{purity_percent:.1f} ± {purity_se_percent:.1f}%',
        'uncertainty_method': 'sqrt(CC_raw); SEM of six raw side peaks; shared-background derivative',
        'background_note': 'Recorded measurement input; its historical estimation algorithm is not in the supplied source.',
    }
    return time, counts, masks, orders, rows, summary


def plot_poster(orders, summary, output_dir):
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = ['#9BBFE0' if k == 0 else '#556F9A' for k in orders]
    heights = summary['normalized_corrected_integrals']
    ax.bar(orders, heights, width=0.78, color=colors, edgecolor='#364A63', linewidth=1.2)
    ax.set_xlabel(r'$\Delta T$ (a.u.)', fontsize=32)
    ax.set_ylabel('Coincidences (a.u.)', fontsize=32)
    # Leave a small gap above 2 so a near-2 central bar does not touch the frame.
    ax.set_ylim(min(0.0, 1.05 * min(heights)), max(2.1, 1.05 * max(heights)))
    ax.set_xticks(orders)
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1e-9, 0.5))
    ax.tick_params(axis='both', labelsize=24, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.text(0.97, 0.92, summary['annotation'], transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='right', va='top')
    fig.tight_layout()
    fig.savefig(output_dir / 'poster_c.png', dpi=300)
    fig.savefig(output_dir / 'poster_c.pdf')
    return fig


def save_integration_check(time, counts, masks, orders, config, output_dir):
    """Separate audit figure; the poster remains the integrated bar chart."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(time, counts, color='#556F9A', linewidth=1.0)
    half_bin = (time[1] - time[0]) / 2
    for mask, order in zip(masks, orders):
        colour = '#9BBFE0' if order == 0 else '#556F9A'
        ax.axvspan(time[mask][0] - half_bin, time[mask][-1] + half_bin, color=colour, alpha=0.25)
        ax.text(np.mean(time[mask]), 1.03 * max(counts[mask]), f'{order:+d}', ha='center', fontsize=11)
    background = config['background_measurement']
    bg_per_bin = background['counts_per_reference_window'] / background['reference_window_bins']
    ax.axhline(bg_per_bin, color='#b22222', linestyle='--', linewidth=1,
               label=f'Recorded background: {bg_per_bin:.1f} per bin')
    ax.set_xlabel('Histogram bin centre (original file coordinate)', fontsize=13)
    ax.set_ylabel('Raw counts', fontsize=13)
    ax.set_title('Shaded windows: bins actually summed for the seven poster bars', fontsize=13)
    ax.set_ylim(0, 1.13 * max(counts))
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / 'g2_integration_check.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=SCRIPT_DIR / 'g2_config.json')
    parser.add_argument('--data', type=Path, help='Override the histogram path in the configuration.')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR)
    parser.add_argument('--no-show', action='store_true', help='Export figures without opening a window.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    data_path = args.data if args.data is not None else args.config.parent / config['data_file']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time, counts, masks, orders, rows, summary = analyze(data_path, config)
    summary['configuration_file'] = str(args.config.resolve())
    with (args.output_dir / 'g2_integrals.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'g2_results.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print('Peak | included bin counts | raw integral | corrected integral | normalized')
    for row in rows:
        print(f"{row['relative_peak']:+d} | {row['bin_counts']} | {row['raw_integral']:.0f} | "
              f"{row['corrected_integral']:.3f} | {row['normalized_corrected_integral']:.6f}")
    print(f"Recorded background = {config['background_measurement']['counts_per_reference_window']:.3f} "
          f"+/- {config['background_measurement']['standard_error_per_reference_window']:.3f} per 3-bin window")
    print(f"CC corrected = {summary['CC_corrected']:.6f}; ACC corrected = {summary['ACC_corrected_mean']:.6f}")
    print(f"g2 = {summary['g2_corrected']:.8f} +/- {summary['g2_standard_error']:.8f}")
    print(summary['annotation'].replace('\n', ' '))
    print('Background is a recorded measurement input, not re-estimated by this script.')
    plot_poster(orders, summary, args.output_dir)
    save_integration_check(time, counts, masks, orders, config, args.output_dir)
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
