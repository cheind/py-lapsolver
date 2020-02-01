import argparse
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Plot benchmark results.', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('benchmarkfile', type=str, help='Json file containing benchmark results')
    #parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    #parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    #parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    #parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

def build_dataframe(args):
    dre = re.compile(r"'(.*)'")

    with open(args.benchmarkfile) as f:
        data = json.load(f)

    events = []
    for b in data['benchmarks']:
        ei = b['extra_info']
        dtype = dre.search(ei['scalar']).group(1)
        events.append(('{}x{}'.format(ei['size'][0], ei['size'][1]), ei['solver'], dtype, b['stats']['mean'], b['stats']['stddev']))

    return pd.DataFrame(events, columns=['matrix-size', 'solver', 'scalar', 'mean-time', 'stddev'])

def draw_plots(df):
    sns.set_style("whitegrid")
    for s, g in df.groupby('scalar'):
        print(g)
        plt.figure()
        title='Benchmark results for dtype={}'.format(s)
        ax = sns.barplot(x='mean-time', y='matrix-size', hue='solver', data=g, errwidth=0, palette="muted")
        ax.set_xscale("log")
        ax.set_xlabel('mean-time (sec)')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('benchmark-dtype-{}.png'.format(s), transparent=True, )
        plt.show()


def main():
    args = parse_args()

    df = build_dataframe(args)
    draw_plots(df)


if __name__ == '__main__':
    main()

