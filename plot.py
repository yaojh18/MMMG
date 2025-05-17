import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap


def plot_model_radar(cat='i'):
    if cat == 'i':
        df = pd.read_csv('./figures/i_eval.csv')
        selected_models = ['GPT Image', 'Gemini Image', 'Imagen 3', 'Dalle 3', 'SD 3.5']
    elif cat == 'a':
        df = pd.read_csv('./figures/a_eval.csv')
        selected_models = ['Stable Audio', 'AudioLDM 2', 'Make-An-Audio 2 (audio only)', 'MusicGen (music only)']
    elif cat == 'it':
        df = pd.read_csv('./figures/it_eval.csv')
        selected_models = ['Gemini Image', 'Gemini 2.5 + Imagen 3', 'GPT-4o + GPT Image', 'Gemini 2.5 + GPT Image']
    elif cat == 'at':
        df = pd.read_csv('./figures/at_eval.csv')
        selected_models = ['Gemini 2.5 + VoxInstruct', 'Gemini 2.5 + VoiceLDM']
    else:
        raise NotImplementedError
    categories = df['task'].tolist()
    N = len(categories)
    offset = np.pi / N
    angles = [(n / float(N) * 2 * np.pi) + offset for n in range(N)]
    angles += angles[:1]

    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'Calibri'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1.2)

    radii = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_rgrids(radii, labels=[''] * len(radii), color='gray', alpha=0.8)
    for r in radii:
        ax.text(0, r, f'{r}', ha='left', va='center', fontsize=18)
    wrapped_categories = ['\n'.join(textwrap.wrap(cat, width=16)) for cat in categories]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=wrapped_categories, fontsize=18)
    for r in radii:
        ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100, '--', color='gray', linewidth=1)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], '--', color='gray', linewidth=1)

    colors = ['#FF5733', '#3357FF', '#659F85', '#A833FF', '#FFBD33']
    for idx, model in enumerate(selected_models):
        values = df[model].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, label=model, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.2)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=18, frameon=True,
               facecolor='white', framealpha=0.8, edgecolor='lightgray', ncol=3)
    plt.tight_layout()
    plt.savefig(f'./figures/{cat}_eval.png', bbox_inches='tight', dpi=600, pad_inches=0.1)


if __name__ == '__main__':
    plot_model_radar('i')
