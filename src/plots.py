import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ['SC', 'SR', 'RPA', 'EA', 'FOTDA-S', 'FOTDA-GL', 'BOTDA-S', 'BOTDA-GL']


def plot_blockwise_accuracy_by_run(csv_file='results/blockwise_accuracy_by_run.csv', 
                                    output_file='results/blockwise_accuracy_by_run.png', verbose=False):
    """Plot accuracy table: methods (rows) × runs (columns), averaged across subjects."""
    df = pd.read_csv(csv_file, index_col=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    for method in df.index:
        row = [method]
        for col in df.columns:
            row.append(f'{df.loc[method, col]:.1f}')
        cell_text.append(row)

    columns = ['Method'] + [f'R{c}' if str(c).isdigit() else c for c in df.columns]

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(df.index)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    run_cols = [i for i, c in enumerate(df.columns) if str(c).isdigit()]
    for j in run_cols:
        col = df.columns[j]
        max_idx = df[col].idxmax()
        max_row_idx = list(df.index).index(max_idx)
        table[(max_row_idx+1, j+1)].set_text_props(weight='bold')
        table[(max_row_idx+1, j+1)].set_facecolor('#90EE90')

    for i in range(len(df.index)):
        table[(i+1, len(columns)-2)].set_facecolor('#FFE6B3')
        table[(i+1, len(columns)-1)].set_facecolor('#FFD6D6')

    plt.title('Cross-Session Block-wise Accuracy (%) - By Run\n(Averaged across subjects)', 
              fontsize=12, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Plot saved to {output_file}")

    plt.close()
    return fig


def plot_blockwise_accuracy_by_subject(csv_file='results/blockwise_accuracy_by_subject.csv',
                                        output_file='results/blockwise_accuracy_by_subject.png', verbose=False):
    """Plot accuracy table: methods (rows) × subjects (columns), averaged across runs."""
    df = pd.read_csv(csv_file, index_col=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    for method in df.index:
        row = [method]
        for col in df.columns:
            row.append(f'{df.loc[method, col]:.1f}')
        cell_text.append(row)

    columns = ['Method'] + list(df.columns)

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(df.index)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    subject_cols = [i for i, c in enumerate(df.columns) if c.startswith('S')]
    for j in subject_cols:
        col = df.columns[j]
        max_idx = df[col].idxmax()
        max_row_idx = list(df.index).index(max_idx)
        table[(max_row_idx+1, j+1)].set_text_props(weight='bold')
        table[(max_row_idx+1, j+1)].set_facecolor('#90EE90')

    for i in range(len(df.index)):
        table[(i+1, len(columns)-2)].set_facecolor('#FFE6B3')
        table[(i+1, len(columns)-1)].set_facecolor('#FFD6D6')

    plt.title('Cross-Session Block-wise Accuracy (%) - By Subject\n(Averaged across runs)', 
              fontsize=12, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Plot saved to {output_file}")

    plt.close()
    return fig


def plot_blockwise_accuracy_evolution(csv_file='results/blockwise_accuracy_by_run.csv',
                                       output_file='results/blockwise_accuracy_evolution.png', verbose=False):
    """Line plot showing accuracy evolution across runs for each method."""
    df = pd.read_csv(csv_file, index_col=0)
    
    run_cols = [c for c in df.columns if str(c).isdigit()]
    df_runs = df[run_cols]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_runs.index)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, method in enumerate(df_runs.index):
        ax.plot(run_cols, df_runs.loc[method], marker=markers[i % len(markers)], 
                label=method, color=colors[i], linewidth=2, markersize=8)

    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Block-wise Accuracy Evolution Across Runs\n(Averaged across subjects)', 
                 fontsize=12, weight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([50, 100])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Plot saved to {output_file}")

    plt.close()
    return fig


def plot_accuracy_table(csv_file='results/blockwise_accuracy_by_subject.csv', output_file='results/accuracy_table.png', verbose=False):
    """Plot accuracy table with methods as rows and subjects as columns."""
    df = pd.read_csv(csv_file, index_col=0)
    
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data. Please run the evaluation first.")
    
    
    if 'Std' in df.columns:
        df = df.drop(columns=['Std'])

    method_order = ['SC', 'SR', 'RPA', 'EA', 'FOTDA-S', 'FOTDA-GL', 'BOTDA-S', 'BOTDA-GL']
    df = df.reindex([m for m in method_order if m in df.index])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    for method in df.index:
        row = [method]
        for col in df.columns:
            row.append(f'{df.loc[method, col]:.2f}')
        cell_text.append(row)

    columns = ['Method'] + list(df.columns)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df.index)))

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(df.index)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for j, col in enumerate(df.columns):
        max_idx = df[col].idxmax()
        max_row_idx = list(df.index).index(max_idx)
        table[(max_row_idx+1, j+1)].set_text_props(weight='bold')

    for i in range(len(df.index)):
        table[(i+1, len(columns)-1)].set_facecolor('#FFE6B3')

    plt.title('Cross-Session Sample-wise Accuracy (%)', fontsize=14, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Table plot saved to {output_file}")

    plt.close()

    return fig


def plot_method_times(csv_file='results/method_times.csv', output_file='results/method_times.png', verbose=False):
    """Plot table of mean times with standard deviation for each method."""
    df = pd.read_csv(csv_file)

    method_order = ['SC', 'SR', 'RPA', 'EA', 'FOTDA-S', 'FOTDA-GL', 'BOTDA-S', 'BOTDA-GL']
    df = df.set_index('Method').reindex([m for m in method_order if m in df['Method'].values]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            row['Method'],
            f"{row['Mean_Time']:.4f}",
            f"{row['Std_Time']:.4f}",
            f"{row['Mean_Time']:.4f} ± {row['Std_Time']:.4f}"
        ])

    columns = ['Method', 'Mean Time (s)', 'Std Dev (s)', 'Summary']

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(len(df)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    min_time_idx = df['Mean_Time'].idxmin()
    table[(min_time_idx+1, 1)].set_facecolor('#90EE90')
    table[(min_time_idx+1, 1)].set_text_props(weight='bold')

    plt.title('Execution Time by Method', fontsize=14, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Time table saved to {output_file}")

    plt.close()

    return fig
