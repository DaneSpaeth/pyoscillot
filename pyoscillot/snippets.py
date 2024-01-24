# Let's try B spline instead
    print(values)
    # tck = bisplrep(x_proj, z_proj, values, kx=3, ky=3)
    
    dx = xx[0,1] - xx[0,0]
    distances_from_origin = np.sqrt(xx**2 + zz**2)
    inside_mask = distances_from_origin <= 1 - np.sqrt(2*(dx/2)**2)
    outside_mask = distances_from_origin > 1 + np.sqrt(2*(dx/2)**2)
    edge_mask = np.logical_and(~inside_mask, ~outside_mask)
    # Also take out the values which are not Nan from the previous interpolation
    edge_mask = np.logical_and(edge_mask, nanmask_grid)
    
    x_edge = xx[edge_mask].flatten()
    z_edge = zz[edge_mask].flatten()
    
    # plt.scatter(x_edge, z_edge)
    # plt.savefig("dbug.png")
    
    # values_interpolated = bisplev(x_edge, z_edge, tck).T
    # print(values_interpolated)
    interp = interp2d(x_proj, z_proj, values, kind="cubic")
    
    # values_interp = interp(x_edge, z_edge)
    # print(values_interp)
    # values_interp.reshape(xx.shape)
    # grid[nanmask_grid] = values_interp[nanmask_grid]