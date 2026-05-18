import json
import argparse
import logging
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Load CMS style including color-scheme (it's an editable dict)
plt.style.use(hep.style.CMS)

if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output",
                        default="limit_plot.png", help='Output file and directory.')
    parser.add_argument('-i', '--input_files', dest='input_files', nargs='+',
                        default=[], help="Json files with limits")
    parser.add_argument('-l', '--labels', dest='labels', nargs='+',
                        default=[], help="Labels for each json file")
    parser.add_argument('-x', '--x-label', dest='x_label',
                        default="r_{HH}", help="Label for the x-axis")
    parser.add_argument('--stat-only', dest='stat_only', action='store_true',
                        help="Only plot the stat. uncertainty (no systematics)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    # Support comma-separated labels passed as a single string
    if len(args.labels) == 1 and ',' in args.labels[0]:
        args.labels = args.labels[0].split(',')


    fig, ax = plt.subplots()

    ticks = []
    ticks_labels = []
    # Create dummy line for legend showing line with marker (skip in stat-only)
    if not args.stat_only:
        ax.plot([], [], color='k', marker='o', markersize=6, linestyle='-', linewidth=1.5, label='observed')
    
    for i, ifile in enumerate(args.input_files):
        limit = json.load(open(ifile, 'r'))['120.0']
        label = args.labels[i] if i < len(args.labels) and args.labels[i] else ifile

        has_obs = 'obs' in limit and not args.stat_only
        ax.vlines( 1, i, i+1, color='tab:red', linestyle='solid' )
        if has_obs:
            ax.vlines(limit['obs'], i, i+1, color='k', linestyle='solid', linewidth=1.5, zorder=10)
            ax.plot([limit['obs']], [i+.5], color='k', marker='o', markersize=6, linestyle='', zorder=11)

        ticks.append( i+.5 )
        obs_line = f"\nObs:{limit['obs']:.2f}" if has_obs else ""
        safe_label = label.replace('_', r'\_')
        ticks_labels.append(f"$\\bf\\mathrm{{{safe_label}}}${obs_line}\nExp:{limit['exp0']:.2f}")
        ax.vlines( limit['exp0'], i, i+0.98, color='k', linestyle='dashed', label=('expected' if i==0 else '') )
        ax.fill_betweenx( [ i, i+0.98], 2*[limit['exp+2']], 2*[limit['exp-2']], color = '#85D1FBff', label=('68% expected' if i==0 else '' ) )
        ax.fill_betweenx( [ i, i+0.98], 2*[limit['exp+1']], 2*[limit['exp-1']], color = '#FFDF7Fff', label=('95% expected' if i==0 else '' )  )

    ax.set_yticks( ticks )
    ax.set_yticklabels( ticks_labels )
    ax.set_xlabel(fr"95% CL limit on $\sigma(\mathrm{{pp}} \to \mathrm{{{args.x_label}}}) / \sigma_{{\mathrm{{Theory}}}}$")
    # ax.set_xlim([0, 10])
    ax.set_ylim([0, len(ticks)+1])
    # hep.cms.label("Supplementary", data=True, loc=1, ax=ax, rlabel="135 fb$^{-1}$ (13 TeV) & 62 fb$^{-1}$ (13.6 TeV)")
    hep.cms.label("Work in Progress", data=True, loc=1, ax=ax, rlabel="135 fb$^{-1}$ (13 TeV)")
    fig.tight_layout()

    # Style
    plt.legend()
    plt.savefig(args.output)
