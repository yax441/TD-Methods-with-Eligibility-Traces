import matplotlib.pyplot as plt
from matplotlib import ticker
from utils.results import flt
import numpy as np

figsize = (7.16,4.43)

def plot_heatmap(v, title=None, labels=True, labels_color='black', labels_size=9.5, xticks=None, yticks=None, cmap='inferno', vmin=None, vmax=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    if len(v.shape) == 1:
        v = v[np.newaxis,:]
    im = ax.imshow(v, cmap=cmap, interpolation='nearest') if vmin is None or vmax is None else ax.imshow(v, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    # ax.axis('off')
    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    if labels:
        # Loop over data dimensions and create text annotations.
        for (j,i),label in np.ndenumerate(v):
            ax.text(i,j,f"{label:.2f}",ha='center',va='center', color=labels_color, size = labels_size)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_episodes(rmse, title=None, xlabel="Episode", ylabel='Average RMSE', ylim=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rmse, linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')
        
def plot_simple_comparison(data_list, labels, title=None, xlabel="Episode", ylabel='Average RMSE', ylim=None, colors = None, legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i,(label,data) in enumerate(zip(labels, data_list)):
        ax.plot(data, label=label, linewidth=1.0) if colors is None else ax.plot(data, label=label, color=colors[i], linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    if title:
        ax.set_title(title,y=0.0, fontsize=10)
    ax.legend() if legend_kwargs is None else ax.legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_best_comparison(results, data_to_compare, alpha_values, lambda_values, n_values, operation='mean', last=None, first=None, skip_first=True,  _min=True, title=None, xlabel="Episode", ylabel='Average RMSE', ylim=None, colors = None, legend_kwargs=None, save_file=None):
    data_list = []
    labels = []
    for a,t in data_to_compare:
        param_values = lambda_values if 'λ' in a else n_values
        best_index = np.argmin(flt(np.array(results[a][t]),operation, last, first, skip_first)) if _min else np.argmax(flt(np.array(results[a][t]),operation, last, first, skip_first))
        best_param_index = best_index//len(alpha_values)
        best_param = param_values[best_param_index]
        best_alpha_index = best_index%len(alpha_values)
        best_alpha = alpha_values[best_alpha_index]
        data_list.append(flt(np.array(results[a][t]),None, last, first, skip_first)[best_param_index][best_alpha_index])
        labels.append(f"{t+' ' if t=='single-update' else ''}{a} (α = {best_alpha:.2f})".replace('λ', f"{best_param}").replace('n-step', f'{best_param}-step'))
    plot_simple_comparison(data_list, labels, title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim, colors = colors, legend_kwargs=legend_kwargs, save_file=save_file)
        
def plot_comparison(rmse_list, line_values, alpha_values, line_prefix, title=None, xlabel='Step size (α)', ylabel='Average RMSE', xlim=None, ylim=None, cmap='gist_rainbow', legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap(cmap,len(rmse_list))
    for i,rmse in enumerate(rmse_list):
        ax.plot(alpha_values,rmse,label=f"{line_prefix}={line_values[i]}", c=colors(i), linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend() if legend_kwargs is None else ax.legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')
    
def plot_parallel_comparison(rmse_lists, alpha_values, line_values, line_prefix, titles, xlabel='Step size (α)', ylabel='Average RMSE', xlim=None, ylim=None, cmap='gist_rainbow', legend_kwargs=None, save_file=None):
    k = len(rmse_lists)
    n = len(rmse_lists[0])
    fig, ax = plt.subplots(k,n,figsize=[x*(max(1.0,k/2) if i else 1.0) for i,x in enumerate(figsize)], sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    if k==1:
        ax = [ax]
    if n==1:
        for i in range(len(ax)):
            ax[i] = [ax[i]]
    for x in range(k):
        colors = plt.cm.get_cmap(cmap,len(rmse_lists[x][0]))
        for i,rmse_list in enumerate(rmse_lists[x]):
            for j,rmse in enumerate(rmse_list):
                ax[x][i].plot(alpha_values,rmse,label=f"{line_prefix}={line_values[j]}", c=colors(j), linewidth=1.0)
            if i>0:
                plt.tick_params('y', labelleft=False)
            else:
                ax[x][i].set_ylabel(ylabel)
                if ylim is not None:
                    ax[x][0].set_ylim(ylim)
            if xlim is not None:
                ax[x][i].set_xlim(xlim)
            ax[x][i].set_title(titles[x][i],y=0.0, fontsize=10)
            ax[x][i].set_xlabel(xlabel)
    ax[0][0].legend() if legend_kwargs is None else ax[0][0].legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_best_and_div_alpha__comparison(rmse_list, x_values, alpha_values, labels, title=None, xlabel="n", colors=None, marker='', log_scale = False, set_ticks = True, ylabel='Average RMSE', ylim=None, legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(2,1,figsize=[x*(1.5 if i else 1.0) for i,x in enumerate(figsize)], sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})
    if title is not None:
        ax[0].set_title(title)
    #best alpha
    ax[0].set_ylabel(ylabel)
    for i,(label,rmse) in enumerate(zip(labels, rmse_list)):
        ax[0].plot(x_values, np.min(rmse, axis=1), marker, label=label, color=colors[i][0], linestyle =colors[i][1], linewidth=1.0) if colors else ax[0].plot(x_values, np.min(rmse, axis=1), marker, linestyle ='solid', label=label, linewidth=1.0)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    #divergence
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Step size (α)')
    ax[1].set_ylim([0.0,1.1])
    for i,(label,rmse) in enumerate(zip(labels, rmse_list)):
        threshold = 5*rmse[0][0]
        indexes = np.argmax(rmse>threshold, axis = 1)
        indexes[indexes==0] = -1
        ax[1].plot(x_values, alpha_values[indexes], marker, label=label, color=colors[i][0], linestyle =colors[i][1], linewidth=1.0) if colors else ax[1].plot(x_values, alpha_values[indexes], marker, linestyle ='solid', label=label, linewidth=1.0)
    if log_scale:
        ax[-1].set_xscale('log')
    if set_ticks:
        ax[-1].set_xticks(x_values)
        ax[-1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax[-1].get_xaxis().set_tick_params(which='minor', size=0)
        ax[-1].get_xaxis().set_tick_params(which='minor', width=0) 
        # ax[0].get_xaxis().set_tick_params(which='minor', size=0)
        # ax[0].get_xaxis().set_tick_params(which='minor', width=0) 
    ax[0].legend() if legend_kwargs is None else ax[0].legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')
        
def plot_best_alpha_comparison(rmse_list, x_values, labels, title=None, xlabel="n", ylabel='Average RMSE', colors=None, marker='', _min = True, log_scale = False, set_ticks = True, ylim=None, legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i,(label,rmse) in enumerate(zip(labels, rmse_list)):
        ax.plot(x_values, np.min(rmse, axis=1) if _min else np.max(rmse, axis=1), marker, label=label, color=colors[i][0], linestyle =colors[i][1], linewidth=1.0) if colors else ax.plot(x_values, np.min(rmse, axis=1) if _min else np.max(rmse, axis=1), marker, linestyle ='solid', label=label, linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if log_scale:
        ax.set_xscale('log')
    if set_ticks:
        ax.set_xticks(x_values)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend() if legend_kwargs is None else ax.legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_div_alpha_comparison(rmse_list, x_values, alpha_values, labels, title=None, xlabel="n", colors=None, marker='', log_scale = False, set_ticks = True, ylim=None, legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i,(label,rmse) in enumerate(zip(labels, rmse_list)):
        threshold = 10*rmse[0][0]
        indexes = np.argmax(rmse>threshold, axis = 1)
        indexes[indexes==0] = -1
        ax.plot(x_values, alpha_values[indexes], marker, label=label, color=colors[i][0], linestyle =colors[i][1], linewidth=1.0) if colors else ax.plot(x_values, alpha_values[indexes], marker, linestyle ='solid', label=label, linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if log_scale:
        ax.set_xscale('log')
    if set_ticks:
        ax.set_xticks(x_values)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Step size (α)')
    if title:
        ax.set_title(title)
    ax.legend() if legend_kwargs is None else ax.legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_best_param_comparison(results_list, alpha_values, labels, title=None, _min=True, alphalim=None, xlabel='Step size (α)', ylabel='Average RMSE', colors=None, xlim=None, ylim=None, legend_kwargs=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    if alphalim is None:
        alphalim = (alpha_values[0],alpha_values[-1])
    alpha_indexes = (alphalim[0]<=alpha_values) & (alpha_values<=alphalim[1])
    for i,(param_values, results) in enumerate(results_list):
        best_index = np.argmin(results)//len(alpha_values) if _min else np.argmax(results)//len(alpha_values)
        ax.plot(alpha_values[alpha_indexes],results[best_index][alpha_indexes],label=labels[i].format(param_values[best_index]), color=colors[i][0], linestyle =colors[i][1], linewidth=1.0) if colors is not None else ax.plot(alpha_values[alpha_indexes],results[best_index][alpha_indexes], linestyle ='solid',label=labels[i].format(param_values[best_index]), linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend() if legend_kwargs is None else ax.legend(**legend_kwargs)
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')
        
def plot_control_comparison(data_list, line_values, alpha_values, line_prefix, interval=None, title=None, ylabel='Average return', xlim=None, ylim=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    color = plt.cm.rainbow(np.linspace(0, 1, len(data_list)))
    for i,data in enumerate(data_list):
        ax.plot(alpha_values,np.mean(data, axis=1),label=f"{line_prefix}={line_values[i]}", c=color[i], linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel('Step size (α)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01,1), loc="upper left")
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_best_alpha_control_comparison(data_list, x_values, labels, interval=None, title=None, xlabel="n", ylabel='Average return', log_scale = False, set_ticks = True, ymax=None, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    for label,data in zip(labels, data_list):
        ax.plot(x_values, np.max(np.mean(data, axis=2), axis=1), label=label, linewidth=1.0)
    if ymax is not None:
        ax.set_ylim([None, ymax])
    if log_scale:
        ax.set_xscale('log')
    if set_ticks:
        ax.set_xticks(x_values)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.47, y-.47), 0.94,0.94, fill=True, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def plot_grid_world(env, title=None, labels_color='black', labels_size=12, cmap='coolwarm', vmin=-0.5, vmax=1.0, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    shape = env.shape
    im = ax.imshow(np.full((*shape,3), fill_value = 255, dtype='B'))
    ax.axis('off')
    # Loop over data dimensions and create text annotations.
    for i in range(shape[1]):
        for j in range(shape[0]):
            r = env.reward[shape[0]-j-1,shape[1]-i-1]
            ax.text(i,j,f"({j+1},{i+1})",ha='center',va='center', color=labels_color, size = labels_size)
            facecolor = "#d4f3fd" if i!=shape[1]-1 or j!=shape[0]-1 else "#ffe6e6"
            edgecolor = "#18c5f4" if i!=shape[1]-1 or j!=shape[0]-1 else "#ff4040"
            if r == -1.0:
                facecolor = '#a6e2f5'
            highlight_cell(i,j, ax, linewidth=2, edgecolor = edgecolor, facecolor = facecolor)
            ax.text(i,j+0.3,f"{r}",ha='center',va='center', size = 9)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')
        
def plot_grid_world_with_policy(avf, title=None, labels_color='black', labels_size=12, save_file=None):
    fig, ax = plt.subplots(figsize=figsize)
    shape = avf.env.shape
    im = ax.imshow(np.full((*shape,3), fill_value = 255, dtype='B'))
    ax.axis('off')
    # Loop over data dimensions and create text annotations.
    symbol = ["↓","→","↑","←"]
    for i in range(shape[1]):
        for j in range(shape[0]):
            r = avf.env.reward[shape[0]-j-1,shape[1]-i-1]
            # ax.text(i,j,f"({j+1},{i+1})",ha='center',va='center', color=labels_color, size = labels_size)
            facecolor = "#d4f3fd" if i!=shape[1]-1 or j!=shape[0]-1 else "#ffe6e6"
            edgecolor = "#18c5f4" if i!=shape[1]-1 or j!=shape[0]-1 else "#ff4040"
            if r == -1.0:
                facecolor = '#a6e2f5'
            highlight_cell(i,j, ax, linewidth=2, edgecolor = edgecolor, facecolor = facecolor)
            # ax.text(i,j+0.3,f"{r}",ha='center',va='center', size = 9)
            # arrows
            if i == shape[1]-1 and j == shape[0]-1:
                continue
            max_value = max(avf.w[j,i])
            position = [
                {'ha': 'center', 'va': 'top', 'x': i, 'y': j-0.06},
                {'ha': 'left', 'va': 'center', 'x': i-0.06, 'y': j},
                {'ha': 'center', 'va': 'bottom', 'x': i, 'y': j+0.12},
                {'ha': 'right', 'va': 'center', 'x': i+0.06, 'y': j}
            ]
            for a in [i for i, val in enumerate(avf.w[j,i]) if val == max_value]:
                ax.text(s=symbol[a], color=labels_color, size='32', **position[a])
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(f'{save_file}.png', dpi=300, bbox_inches='tight')

def plot_grid_world_optimal_policy(avf, title=None, labels_color='black', save_file=None):
    optimal_action = np.full(avf.env.shape, None).tolist()
    optimal_action[0][0] = [0,1]
    m,n = avf.env.shape
    for y in range(n):
        if y<avf.env.shape[1]-1:
            optimal_action[0][n-1-y] = [1]
        optimal_action[1][y] = [0]
        optimal_action[3][n-1-y] = [0]
        optimal_action[4][y] = [1]
    optimal_action[1][n-2] = [0,1]
    for x in range(m):
        optimal_action[x][avf.env.shape[1]-1] = [0]
        if x>0 and x<m-1:
            optimal_action[x][0] = [0]
    for y in range(1,n-1):
        optimal_action[2][y] = [3] if (y+1)-10<1-(y+1) else [0,1]
        if y-10 == 1-y:
            optimal_action[2][y].append(3)
    for x in range(m):
        for y in range(n):
            for a in optimal_action[x][y]:
                avf.w[x,y,a] = 1.0
    plot_grid_world_with_policy(avf, title=title, labels_color=labels_color,  save_file=save_file)