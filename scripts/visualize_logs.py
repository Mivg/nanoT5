import re
import plotly.graph_objects as go  # require pip install plotly
import pandas as pd
from typing import List, Tuple
import sys

def parse_log_file(file_path: str, key: str = 'Loss') -> pd.DataFrame:
    """
    Parses a log file and extracts the specified key's values along with step numbers and phase (train/eval).

    :param file_path: Path to the log file.
    :param key: The key to extract values for (e.g., 'Loss', 'Weights_l2', 'Grad_l2', 'Lr',
    'Seconds_per_step', 'eta', 'rbar', 'G').
    :return: DataFrame with columns ['Step', 'Value', 'Phase'].
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expression pattern to match the log line format and extract relevant data
    pattern = r'\[([0-9-]+ [0-9:,]+)\]\[([A-Za-z]+)\]\[INFO\] - \[([a-z_]+)\] Step (\d+) out of \d+ \| .*?{} --> ([\d.]+) \|'.format(re.escape(key))

    data = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(4))
            value = float(match.group(5))
            phase = match.group(3)
            data.append((step, value, phase))

    return pd.DataFrame(data, columns=['Step', 'Value', 'Phase'])

def plot_data(files: List[Tuple[str, str]], key: str = 'Loss') -> str:
    """
    Plots the data from multiple log files.

    :param files: List of tuples (file_path, name) for each log file.
    :param key: The key to plot.
    :return: Path to the saved plot HTML file.
    """
    fig = go.Figure()
    PHASES = ['train', 'eval', 'eval_av']
    PHASES_MARKERS = dict(zip(PHASES, ['circle', 'square', 'x']))

    for name, file_path in files:
        df = parse_log_file(file_path, key)
        for phase in ['train', 'eval', 'eval_av']:
            phase_df = df[df['Phase'] == phase]
            if len(phase_df) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=phase_df['Step'],
                y=phase_df['Value'],
                mode='lines+markers',
                name=f"{name} [{phase}]",
                marker=dict(
                    symbol=PHASES_MARKERS[phase]
                )
            ))

    fig.update_layout(
        title=f"{key} per Step",
        xaxis_title="Step",
        yaxis_title=key,
        legend_title="Log Files"
    )

    output_file = './plot.html'
    fig.write_html(output_file)

    return output_file


if __name__ == '__main__':
    # Example usage
    file_paths = [f.split('=') for f in sys.argv[1:]]
    output_plot = plot_data(file_paths, key='Lr')
    # output_plot = plot_data(file_paths, key='Loss')
    print(f"Plot saved to {output_plot}")

