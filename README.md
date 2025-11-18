# k-Core Visualizations
This repository contains code for creating the network visualizations in the paper *Deterministic Construction of Typical Networks in Network Models* and a brief tutorial on how to use it. The visualization algorithm is based on [LaNetVI](https://en.wikipedia.org/wiki/LaNet-vi) by [Alvarez-Hamelin et al. (NeurIPS, 2005)](https://papers.nips.cc/paper_files/paper/2005/hash/b19aa25ff58940d974234b48391b9549-Abstract.html).


## Installation
Before using `kcore-viz`, you will need to install `graph-tool` as explained [here](https://graph-tool.skewed.de/installation.html). If you are using `conda` you can

```bash
conda create -n vizenv python=3.11
conda install -c conda-forge graph-tool=2.97
conda activate vizenv
```

Then you can directly `pip install` from this repository using

```bash
pip install "git+https://github.com/moritz-laber/kcore-viz"
```

If you want to experiment with the tutorial, you might want to clone the repository and install additional dependencies via

```bash
git clone https://github.com/moritz-laber/kcore-viz
cd kcore-viz
pip install -e ".[tutorial]"
```

## Usage
We provide access to the visualization algorithm used in the paper through the function `draw_kcore_viz`, a minimal example is shown bellow

```python
from kcoreviz import draw_kcore_viz
import graph_tool as gt
from graph_tool import *

# load the Karate Club network from the Netzschleuder repository
g = gt.collection.ns["karate/77"]

# create the visualization
draw_kcore_viz(
    g,
    output="./karate.svg"
)
```

For a more detailed description, and ways to customize the layout see the `/tutorial/tutorial.ipynb`. There, we also explain how the functions `shell_cluster_decomposition`, `vertex_positions`, `linscaling`, and `edge_filter` can be used to create similar layouts that are based on network properties other than corness, e.g., degree.