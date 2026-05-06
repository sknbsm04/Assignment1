import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import plotly.graph_objects as go
import pandas as pd
from scipy.io import loadmat

def get_data():
    
    data0 = pd.read_csv('data/data0.csv', index_col=0).to_numpy()
    data1 = pd.read_csv('data/data1.csv', index_col=0).to_numpy()
    data2 = pd.read_csv('data/data2.csv', index_col=0).to_numpy()
    data3d = pd.read_csv('data/data3d.csv', index_col=0).to_numpy()
    bones = loadmat('data/shapes.mat')['aligned']

    print(f'data0 shape: {data0.shape}\ndata1 shape: {data1.shape}\ndata2 shape: {data2.shape}\ndata3d shape: {data3d.shape}\nbones shape: {bones.shape}')

    return data0, data1, data2, data3d, bones


def plot2DPCA(data, mju, eigVec, eigVal, recData=None, showStd=None, showReconstruction=None):
    """
    Plot PCA results in 3D using matplotlib.
    
    Parameters:
        data: (n_samples, 2) original data
        mju: (2,) mean vector
        eigVec: (2, 2) eigenvectors
        eigVal: (2,) eigenvalues
        recData: (n_samples, 2) reconstructed data (optional)
        showReconstruction: bool, whether to show reconstruction
    """

    fig, ax = plt.subplots()
    
    # Plot original data
    ax.plot(data[:, 0], data[:, 1], 'k.', label='data')

    # Mean
    ax.plot(mju[0], mju[1], 'ro', label='mean')

    # Principal directions
    v1 = eigVec[:, 0]
    e1 = eigVal[0]
    dir1 = v1 * np.sqrt(e1)

    v2 = eigVec[:, 1]
    e2 = eigVal[1]
    dir2 = v2 * np.sqrt(e2)

    ax.plot([mju[0]-dir1[0], mju[0]+dir1[0]], [mju[1]-dir1[1], mju[1]+dir1[1]], 'r-', label='eigenvectors')
    ax.plot([mju[0]-dir2[0], mju[0]+dir2[0]], [mju[1]-dir2[1], mju[1]+dir2[1]], 'r-')

    # Covariance matrix
    C = np.cov(data, rowvar=False, bias=True)

    if showStd == 1:
        # Plot confidence ellipses
        def plot_ellipse(conf, color, label):
            chi2_val = chi2.ppf(conf, df=2)
            vals, vecs = np.linalg.eigh(C)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(chi2_val * vals)
            ell = Ellipse(xy=mju, width=width, height=height, angle=theta,
                          edgecolor=color, linestyle='--', facecolor='none', label=label)
            ax.add_patch(ell)

        plot_ellipse(0.683, 'r', '1x standard deviation')
        plot_ellipse(0.954, 'g', '2x standard deviation')
        plot_ellipse(0.997, 'b', '3x standard deviation')

    if showReconstruction == 1:
        ax.plot(recData[:, 0], recData[:, 1], 'g*', label='Reconstruction')
        for i in range(len(data)):
            ax.plot([data[i, 0], recData[i, 0]], [data[i, 1], recData[i, 1]], ':', color='gray')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=3, borderaxespad=0., frameon=False)
    plt.show()


def plot3DPCA(data, mju, eigVec, eigVal, recData=None, showReconstruction=None):
    """
    Plot PCA results in 3D using Plotly.
    
    Parameters:
        data: (n_samples, 3) original data
        mju: (3,) mean vector
        eigVec: (3, 3) eigenvectors
        eigVal: (3,) eigenvalues
        recData: (n_samples, 3) reconstructed data (optional)
        showReconstruction: bool, whether to show reconstruction
    """
    traces = []

    # 1. Original data
    traces.append(go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.5),
        name='Daten'
    ))

    # 2. Mean
    traces.append(go.Scatter3d(
        x=[mju[0]], y=[mju[1]], z=[mju[2]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Mittelwert'
    ))

    # 3. Principal directions
    for i in range(3):
        vec = eigVec[:, i] * np.sqrt(eigVal[i])
        start = mju - vec
        end = mju + vec

        traces.append(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(width=8),
            name=f'PC{i+1}'
        ))

    # 4. Reconstruction (if any)
    if showReconstruction and recData is not None:
        traces.append(go.Scatter3d(
            x=recData[:, 0], y=recData[:, 1], z=recData[:, 2],
            mode='markers',
            marker=dict(size=1, color='green', symbol='x'),
            name='Reconstruction'
        ))

        # Connecting lines from original to reconstructed
        for i in range(len(data)):
            traces.append(go.Scatter3d(
                x=[data[i, 0], recData[i, 0]],
                y=[data[i, 1], recData[i, 1]],
                z=[data[i, 2], recData[i, 2]],
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                showlegend=False
            ))

    # Final layout and plot
    layout = go.Layout(
        width=800,
        height=800,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, b=20, t=20)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    

def plotDemo():

    eigVal = np.load('demo/eigVal.npy')
    eigVec = np.load('demo/eigVec.npy')
    mean = np.load('demo/mean.npy')
    data = pd.read_csv('demo/demo.csv', index_col=0).to_numpy()

    # Plotting the PCA
    plot2DPCA(
        data, 
        mean, 
        eigVec, 
        eigVal, 
        showStd=True,
        #recData=redData, # here goes your reconstructed data 
        showReconstruction=False)