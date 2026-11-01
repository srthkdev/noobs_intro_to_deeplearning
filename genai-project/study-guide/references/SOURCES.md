# Research reference figures

The reference figures introduce the architecture ideas. They are identified separately from our checkpoint-derived diagrams and original project outputs. Credit belongs to their respective authors.

- **U-Net, Figure 1**: Olaf Ronneberger, Philipp Fischer and Thomas Brox (2015), [paper](https://arxiv.org/abs/1505.04597), [authors’ project page](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). PNG and vector SVG show the figure region of paper page 2, extracted without changing the diagram. The original 572 × 572 input and two-class output differ from our implementation.
- **ResNet, Figure 2**: Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (2015), [paper](https://arxiv.org/abs/1512.03385). PNG and vector SVG show the residual-block figure on paper page 2. This is the residual shortcut inside an encoder, not a long U-Net skip.
- **FPN, Figure 3**: Tsung-Yi Lin and coauthors (2017), [paper and figure context](https://arxiv.org/html/1612.03144v2#S3), [original image](https://arxiv.org/html/1612.03144v2/block.png). The unmodified PNG explains a lateral addition; our mask-producing FPN differs from the paper’s detection system.

The guide labels its own diagrams as schematic, lists actual tensor shapes and explains implementation differences alongside these references.

Notebook rendering uses [Jupyter nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html), with [Pygments](https://pygments.org/) for syntax highlighting. Source files are rendered without executing cells. Markdown, saved image outputs, text and dataframe tables come from the original notebooks. Separately saved artifacts are identified as such.
