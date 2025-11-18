### IMPORTS ###
import graph_tool as gt
from graph_tool.all import *
from matplotlib.colors import Colormap, to_rgba, to_hex
from matplotlib import colormaps
import numpy as np
from typing import List, Tuple, Union

### UTILITY FUNCTIONS ###
def linscaling(g:gt.Graph, property:gt.PropertyMap, s0:float, s1:float, reverse=False)->gt.PropertyMap:
    """Compute a linearly-scaled property map for the input graph based on the input property map.
       If p is a property than the linearly-scaled property s is defined as
       s[v] = (s1 - s0) * (p[v] - pmin)/(pmax - pmin) + s0,
       where pmin and pmax are the minimum and maximum values of the property map p.

    """

    assert s1 >= s0 >= 0, "Bad sizes. The sizes need to satisfy s1 >= s0 >= 0."

    scale = g.new_vp('double')
    pmin  = np.min(property.a)
    pmax  = np.max(property.a)
    if reverse:
        for v in g.vertices(): scale[v] = (s1 - s0) * (1.0 - (property[v] - pmin)/(pmax - pmin)) + s0
    else:
        for v in g.vertices(): scale[v] = (s1 - s0) * (property[v] - pmin)/(pmax - pmin) + s0

    return scale

def edge_filter(g:gt.Graph, efrac:float, rng:np.random.Generator=np.random.default_rng())->gt.GraphView:
    """Create a graph view of the input graph by randomly removing edges.

    Input
    g : the input graph
    efrac : fraction of edges to keep. Must be in [0, 1].

    Output
    g_filtered : a graph view with a random subset of edges.
    """

    assert 0.0 <= efrac <= 1.0, "efrac needs to be between 0 and 1."

    efilt = g.new_ep('bool')
    for e in g.edges(): efilt[e] = rng.random()<efrac
    g_filtered = gt.GraphView(g, efilt=efilt)

    return g_filtered

def shell_cluster_decomposition(g:gt.Graph, property:gt.PropertyMap)->Tuple[dict, dict]:
    """Decompose the input graph into shells (i.e., vertex induced subgraphs with the same property value)
       and clusters (i.e., connected components within a shell).

    Input 
    g : the input graph
    property : the property map used to define shells

    Output
    shells : dictionary mapping property values to graph views of the corresponding shells
    clusters : dictionary mapping property values to property maps labeling the connected components within the corresponding shells
    """

    # compute shells
    shells = {p : gt.GraphView(g, vfilt=[True  if property[v]==p else False for v in g.vertices()]) for p in np.unique(property.a)}
    
    # compute clusters within shells
    clusters = {p : gt.topology.label_components(shells[p])[0] for p in np.unique(property.a)}

    return shells, clusters

def vertex_positions(g:gt.Graph, property:gt.PropertyMap, clusters:dict, eps:float=0.85, r0:float=1.0, r1:float=10.0, rng:np.random.Generator=np.random.default_rng()) -> gt.PropertyMap:
    """Compute the vertex positions where the property value determines the radial coordinate
       and the cluster membership determines the angular coordinate.
    
    Input
    g : the input graph
    property : the property map used to determine radial positions
    clusters : dictionary mapping property values to property maps labeling the connected components within a shell of same property value.
    eps : mixing parameter in [0, 1] controlling how much a nodes radial coordinate depends on its own property value (eps=1) vs the average property value of its neighbors (eps=0).
    r0 : minimum radial coordinate
    r1 : maximum radial coordinate
    rng : a numpy random number generator

    Output
    pos : a property map of vertex positions in 2D Cartesian coordinates.
    """

    # checks
    assert 0.0 <= eps <= 1.0, "eps needs to be between 0 and 1."

    assert r1 > r0 > 0, "Bad radii. The radii need to satisfy r1 > r0 > 0."

    # determine the sizes of the clusters
    size_clusters = {p : {q : 0 for q in np.unique(clusters[p].a)} for p in np.unique(property.a)}
    for v in g.vertices():
        p = property[v]
        q = clusters[p][v]
        size_clusters[p][q] +=1

    cum_size_clusters = {p : {q : np.sum([size_clusters[p][q] for q in range(q)]) for q in np.unique(clusters[p].a)} for p in np.unique(property.a)}
    total_size_clusters = {p : np.sum([s for s in size_clusters[p].values()]) for p in np.unique(property.a)}
    
    # compute the radial positions
    cprime = g.new_vertex_property("double")
    for v in g.vertices():
        cprime[v] = eps * property[v] + (1.0 - eps) * np.mean([property[u] for u in g.iter_all_neighbors(v)])
    r = linscaling(g, cprime, r0, r1, reverse=True)

    # compute the angular positions
    phi = g.new_vertex_property("double")
    for v in g.vertices():
        p = property[v]
        q = clusters[p][v]

        # determine the radial slice and place vertices uniformly within it
        phi_min = 2*np.pi*(cum_size_clusters[p][q] - size_clusters[p][q])/total_size_clusters[p]
        phi[v] = phi_min + rng.random() * size_clusters[p][q]

    # change to Cartesian coordinates
    pos = g.new_vertex_property("vector<double>")
    for v in g.vertices():
        pos[v] = [r[v] * np.cos(phi[v]), r[v] * np.sin(phi[v])]

    return pos

### K-CORE LAYOUT ###
def draw_kcore_viz(
    g0:gt.Graph,
    cmap:Union[Colormap,str]=colormaps['viridis'],
    eps:float=0.85,
    r:Tuple[float]=(2.0, 25.0),
    s:Tuple[float]=(2.0, 20.0),
    w:Tuple[float]=(0.1, 1.2),
    vedge_color:Union[str, Union[List[float], Tuple[float]]]='#343837',
    efrac:float=1.0,
    ecolor:Union[str, Union[List[float], Tuple[float]]]=[0.847, 0.863, 0.839, 0.95],
    fit_view:bool=True,
    output_size:Tuple[float]=(800, 800),
    seed:int=1729,
    output:str='./network_viz.svg'
    )->None:
    """Create a visualization of the largest connected component of the input graph using the k-core layout, a basic version of LaNet-vi.
    
    Input
    g0 : the input graph
    cmap : a colormap (as a callable or matplotlib.colors.Colormap)
    eps : real parameter in [0,1] that controls how much a nodes radial coordinate depends on its own coreness (eps=1) vs the average coreness of its neighbors (eps=0).
    r : tuple of the minimum and maximum radius at which nodes are placed.
    s : tuple of the minimum and maximum size of nodes.
    w : scale for width of vertex borders.
    vedge_color : color for vertex borders.
    efrac : fraction of visualized edges. Must be in [0, 1].
    ecolor : color of the edges. Must be in rgba format.
    fit_view : whether to rescale the visualization to the provided size.
    output_size : size of the visualization.
    seed : seed for the pseudo-random number generator. 
    output : path to output file.

    Output
    None - creates a file containing the visualization in the output directory.
    """

    # unpack parameters
    r0, r1 = r
    s0, s1 = s
    w0, w1 = w

    # checks
    if not (r1 > r0 > 0):
        raise ValueError("Bad radii. The radii need to satisfy r1 > r0 > 0.")
    if not (s1 >= s0 > 0):
        raise ValueError("Bad sizes. The sizes need to satisfy s1 >= s0 > 0.")
    if not (w1 >= w0 > 0):
        raise ValueError("Bad widths. The widths need to satisfy w1 >= w0 > 0.")
    if not (0.0 <= efrac <= 1.0):
        raise ValueError("efrac needs to be between 0 and 1.")
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps needs to be between 0 and 1.")

    if isinstance(cmap, str):
        try:
            cmap = colormaps[cmap]
        except KeyError:
            raise ValueError(f"Unknown colormap: {cmap}")
    
    if isinstance(ecolor, List) or isinstance(ecolor, Tuple):
        if len(ecolor) == 3:
            ecolor = [*ecolor, 1.0]
        elif len(ecolor) != 4:
            raise ValueError("Edge color must be a string or a list/tuple of length 3 (rgb) or 4 (rgba).")
    elif isinstance(ecolor, str):
        ecolor = to_rgba(ecolor)
    else:
        raise ValueError("Edge color must be a string naming a color or a list/tuple of length 3 (rgb) or 4 (rgba).")
    
    if isinstance(vedge_color, List) or isinstance(vedge_color, Tuple):
        if len(vedge_color) == 3:
            vedge_color = to_hex([*vedge_color, 1.0])
        elif len(vedge_color) == 4:
            vedge_color = to_hex(vedge_color)
        else:
            raise ValueError("Vertex edge color must be a string or a list/tuple of length 3 (rgb) or 4 (rgba).")
    elif isinstance(vedge_color, str):
        pass
    else:
        raise ValueError("Vertex edge color must be a string naming a color or a list/tuple of length 3 (rgb) or 4 (rgba).")
        

    # seed the random number generator
    rng = np.random.default_rng(seed=int(seed))

    # extract largest connected component
    g = gt.topology.extract_largest_component(g0, prune=True)

    # compute the k-core decomposition
    coreness = gt.topology.kcore_decomposition(g)

    # compute the vertex degrees
    degree = g.degree_property_map('total')

    # determine the clusters, i.e., connected components within shells of same coreness
    _, clusters = shell_cluster_decomposition(g, coreness)

    # compute the vertex positions
    pos = vertex_positions(g, coreness, clusters, eps, r0, r1, rng)

    # determine vertex size
    log_degree = g.new_vertex_property("double")
    gt.map_property_values(degree, log_degree, lambda x: np.log(x) if x > 0 else 0)
    vsize = linscaling(g, log_degree, s0, s1)

    # compute the width of the vertex borders
    vwidth = linscaling(g, degree, w0, w1)
    
    # compute the vertex colors
    coreness_norm = linscaling(g, coreness, 0.0, 1.0)
    vcolor = g.new_vp('vector<double>')
    for v in g.vertices(): vcolor[v] = cmap(coreness_norm[v])

    # subset edges
    g_draw = edge_filter(g, efrac, rng)

    # draw the graph
    gt.draw.graph_draw(
        g_draw,
        pos=pos,
        vprops={'size' : vsize, 'fill_color' : vcolor, 'color' : vedge_color, 'pen_width': vwidth},
        eprops={'color':ecolor},
        fit_view=fit_view,
        output_size=output_size,
        output=output
    )

