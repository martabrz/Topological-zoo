import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def latex_plot(scale = 1, fontsize = 12):
    """ Changes the size of a figure and fonts for the publication-quality plots. """
    fig_width_pt = 246.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = fig_width*golden_mean
    fig_size = [fig_width, fig_height]
    eps_with_latex = {
        "pgf.texsystem": "pdflatex", "text.usetex": True, "font.family": "serif", "font.serif": [], "font.sans-serif": [], "font.monospace": [], "axes.labelsize": fontsize, "font.size": fontsize, "legend.fontsize": fontsize, "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.figsize": fig_size
        }
    mpl.rcParams.update(eps_with_latex)

def plot_energy_states(evals, name, mode = 'line', scale = 0.9):
    # Plots the eigenvalues; mode = 'line' or 'normal'.
    latex_plot(scale)
    N = len(evals)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('E')
    ax.set_ylim(np.floor(np.min(evals)), np.ceil(np.max(evals)))
    if mode == 'line':
        x1 = np.empty(N)
        x2 = np.empty(N)
        x1.fill(0.5)
        x2.fill(1.5)
        ax.plot([x1, x2], [evals, evals], linestyle = '-', color = 'k', linewidth = 0.5)
        ax.set_xticks([])
        ax.set_xlabel('')
    elif mode == 'normal':
        ax.plot(np.arange(1, N + 1), evals, 'ko')
        ax.set_xlabel('eigenvalue index')
    fig.savefig(name + '.svg', transparent = True, dpi = 800, bbox_inches = 'tight')

def plot_bands1D(k, evals, name, scale = 0.9):
    # Plots the 1D band structure E = f(k).
    latex_plot(scale)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('k')
    ax.set_ylabel('E')
    ax.set_xlim(np.min(k), np.max(k))
    ax.set_ylim(np.floor(np.min(evals)), np.ceil(np.max(evals)))
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels([0, r'$\pi / 2$', r'$\pi$', r'$3 \pi / 2$', r'$2 \pi$'])
    for i in range(evals.shape[1]):
        ax.plot(k, evals[:, i], 'k')
    fig.savefig(name + '.svg', transparent = True, dpi = 800, bbox_inches = 'tight')

def plot_localization1D(evec, name, scale = 0.9):
    # Plots the localization of a given eigenstate 'evec'.
    latex_plot(scale)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('site index')
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_ylim(0, 1)
    N = len(evec)
    psi2 = np.abs(evec**2)
    if not np.isclose(np.sum(psi2), 1):
        raise ValueError('Amplitudes do not sum to 1, check your eigenvector!')
    ax.plot(np.arange(1, N + 1), psi2, 'ko-', markersize = 2)
    fig.savefig(name + '.svg', transparent = True, dpi = 800, bbox_inches = 'tight')
