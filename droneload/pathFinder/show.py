def draw_path_plt(ax, path):
    x, y, z = path[0,:], path[1,:], path[2,:]
    ax.scatter([path[0, 0]], [path[1, 0]], [path[2, 0]], c='r', marker='o')
    ax.plot(x, y, z)