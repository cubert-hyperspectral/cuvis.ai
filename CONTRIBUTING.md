# Contributing to cuvis.ai

First off, thank you for considering making contributions to cuvis.ai!
This document should give you all the information necessary to make a meaningful contribution to the cuvis.ai project.
Below, [General Process](#general-process) outlines the process of writing, testing, pushing and merging a code contribution.
Section ["Making a Contribution"](#making-a-contribution) provides additional information, depending on the kind of contribution you aim to make.
If you plan on providing a dataset you own the rights to to cuvis.ai, you can skip to section [Contributing Datasets](#contributing-datasets).

## General Process

For all code contributions, we suggest you follow this general process.

### Coding Guidelines

Any code you write for cuvis.ai should follow these guidelines.
We format all code according to [PEP8](https://peps.python.org/pep-0008/), so you should too.
It is easy to let a formatter run before you push your code or simply add a "format on save" setting to the editor of your choice.

Further style choices:
 - All indentation uses spaces only, 4 spaces per indentation
 - Method names for methods for class-internal use only are prefixed with an underscore
 - CamelCase for class names, snake_case for everything else

When writing a new Node, please read the source code for the base Node class.
You can find that in the repository, under: cuvis_ai/node/node.py
Every abstract method of Node must be implemented. Most nodes also have a *fit()* method.

**TODO**: Explain dimensionality verification

#### Repository Structure

Algorithms / Nodes are sorted into 6 categories:
 - *Transformation* nodes only apply simple mathematical (eg. normalization) or geometric (eg. resize) operations on input data. They may also transform the labels or metadata
 - *Preprocessor* nodes apply more complex operations (eg. PCA) on input data, but they do not have any internal state that needs to be trained or fitted to the data
 - *Unsupervised* nodes contain algorithms (eg. KMeans) that require a training or fit step to initialize their internal state
 - *Supervised* nodes contain algorithms (eg. SVM) cannot train their internal state using just input  data alone, they also require labels
 - *Distance* nodes contain comparison algorithms (eg. Euclidean Distance) that can compare an example spectrum / data point to the rest of the  data and provide a similarity / distance measure per data point
 - *Deciders* These nodes take continuous results and discretize them. Such a node usually is the final step that provides the result for a graph (eg. BinaryDecider)

Further directories include:

 - *data:* Contains code for loading datasets and other data utilities
 - *node:* Contains the base abstract node class and other abstract classes
 - *pipeline:* Contains the Graph class
 - *test:* Contains all unit tests
 - *tv_transforms:* Contains extensions to the pytorch torchvision transforms project used in cuvis.ai
 - *utils:* Contains generic utilities used throughout cuvis.ai


### Documentation Guidelines

TODO

### Making a Pull-Request

TODO

## Making a Contribution

We need to differentiate between making contributions to the core code of the framework, contributing new algorithms (Nodes) and contributing data sets, all of which are very welcome.

### Contributing new Algorithms / Nodes

Before you implement and contribute a new algorithm, please make sure that it is not patented and that you are legally allowed to contribute it to another project.
Check that either you own the copyright or the software license it is under allows for reuse under the Apache 2.0 license.
Please make sure to fulfill all obligations that the license, if any, require.

Choose the correct category for your node, see section [Repository Structure](#repository-structure).
If you are not sure which category is the correct one, feel free to contact us: [Further Questions](#further-questions)

Name your feature branch correctly: "contrib_node/[my_node_name]"

### Contributing Core-Code

Nothing extra to mention here, just name your feature branch correctly: "contrib_code/[my_node_name]"

### Contributing Datasets

TODO

## Further Questions

If you're unsure about something or just have a question in general, feel free to open an issue on GitHub about it.
Before you open a new issue, use the search feature on the [issues page](https://github.com/cubert-hyperspectral/cuvis.ai/issues) and see if there already is an existing issue that pertains to your question or concern.
If your search leaves you empty handed, create a new issue, give it a **suitable name** and **label** and write a meaningful description.
If you feel like the issues page is not the correct avenue for communication, you will find more ways to get in touch with us on our [website](https://www.cubert-hyperspectral.com/).