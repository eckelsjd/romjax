## Goal
Design a method that auto-generates a set of GraphLossTerms based on some simple configurations and passes all the terms to a GraphLoss function that can be used online with the Train routine for a FunctionGraph. You will first need to support the ability to auto-generate terms in the GraphLoss configuration.

Then, design a specific term-generator method that uses a cyclical FunctionGraph of N nodes and produces N*N terms according to the following specs:

- Each term is the graph path error between the clockwise and counter-clockwise paths around the cyclical graph from node_i->node_j. You can imagine this forming an NxN matrix, where element ij is the || cw - ccw || path error between nodes i and j. Each node defines the specific error operation, so you should compute the error of each term using the destination node j.
- The user can either provide the ordering of the nodes, or they can be inferred from the FunctionGraph (will assume/check that these nodes do in fact form a compatible cycle). Assume the nodes are listed in clockwise order.
- The names for each term should be generated as {node_i}->{node_j}, using the names of the nodes themselves
- All terms will use the same user-specified weight, dataset, batch_reduce, and batch_size
- The dataset name specifies which edge in the graph that dataset originates from, e.g. "poisson" for Poisson2D. You should assume that the passed-in data contains all data relevant to both nodes corresponding to this edge, e.g. the source and target nodes of Poisson2D collectively contain a data tree of the form {inputs: ..., outputs: ..., residuals: ...}.
- Any path starting from either the source or target nodes of the dataset edge can skip the forward/backward evaluation of that edge, since the data already contains the results. The only exception is the very last step of a full cycle path, where the dataset edge should be evaluated once in the corresponding direction to bring a cycle-transformed payload back to the starting node. For example, i->j->k->l->i should only evaluate l->i, assuming the i node corresponds to one of the dataset nodes.
- Any node that is not the source or target of the dataset-producing edge must be "seeded" first by pushing the data along one path from the dataset nodes to the target "outside" node, before any of the path errors can be computed. For this, push the dataset payload along the shortest path to the target node. For example, if the dataset edge is i->l and the ordered graph is i->j->k->l->i, then seeding node j means pushing along i->j, and seeding node k means pushing l->k. This is because node l is part of the dataset edge and is closer to k than node i.
- Each term should optionally accept the same template_paths and aux_paths supplied by the user to the generator function -- these are used to specify any auxiliary templates need by edges in the graph ahead of time, when they are not possible to infer internally, e.g. via eqx_evaluate. This is the same way reconstruction_loss currently handles templates.
- The user can optionally specify whether auxiliary data production/passing is allowed between the generated terms. The tradeoff is compute versus memory. For small problems, caching intermediate graph payloads can reduce duplicate operations between terms. For large problems, this may not be possible to store so an extra compute cost is incurred.
- You will need to design and implement an appropriate caching format for passing intermediate graph payloads between graph terms. All the graph loss terms should know how to produce and use this cache from auxiliary data. The data that should be cached is the result of passing a payload from a start node along a certain path to a target node. Then, downstream nodes that partially follow that same path can use the cached payload rather than recomputing it. For example, if the dataset edge is i->l in the i->j->k->l->i graph, the i->i error term will first compute || (i->j->k->l->i) - (i->l->k->j->i) ||, which has 6 intermediate terms: i->j, i->j->k, i->j->k->l, and i->l, i->l->k, i->l->k->j. Then, there are several other error terms that follow some of these same paths, e.g. the i->l error term will use the i->l path and the i->j->k->l path, which have already been computed. Note that the seeded terms will need to include the initial seeding path in how they describe their cached payloads, e.g. seeding i->l and then taking the l->i path should look like data that started at i then took the path i->l->i
- For now, any internal auxiliary data produced by edges during the internal graph push path mechanism should be assumed only useful for a single data and a single path error term. You should reuse this aux data within a single term evaluation as you traverse the relevant paths, but you should not pass this internal aux data between separate graph loss terms.
- Your design should ideally integrate cleanly with the existing graph.path_error method. You are allowed to edit graph.path_error to support cached path payloads if needed, or if it is simpler you can manage all the caching logic in the graph loss terms. Note that you will somehow need to implement the logic for taking single steps around specified paths in a graph, using cached data when available. If caching is disabled, then graph.path_error will likely work just fine as it is currently written.
- If possible, brainstorm some ways that intermediate payload caching memory can be improved. For example, it should be possible to detect ahead of time what paths each term is going to need to take. You can then build an optimal ordering of terms so that downstream terms can remove items from the cache when they know they are the last one that needs it. Or the user can specify which graph edges are the most expensive, and only cache a certain number of paths that traverse those edges.

## Configuration requirements
The auto-generation function can be configured as a CallableModel from yaml and passed directly as a set of terms to the loss function in the Train routine, e.g. something like:
```yaml
!romx:Train
loss: 
  terms:
    - callable: gen_terms                               # detects and expands to multiple GraphLossTerms
      paths: etc...
      opts: other opts for the gen function...  
    - term: { callable: reconstruction, path: [...] }   # other terms can still be passed in manually
      weight: 1.0
      batch_reduce: mean
    - {callable: tikhonov }  # this should still validate to GraphLossTerm(term={callable: tikhonov}) for example
    - callable: other_gen_terms                         # can even generate multiple times with different generators
```

## Your plan details
Your design and implementation plan should specifically include details on the following items:

- How you will integrate auto term-generation into the existing yaml/GraphLoss configuration workflow
- How you will generate graph loss term callables for the cyclic graph path errors, including how you will detect and use the clockwise and counter-clockwise paths
- How you will handle dataset-nodes versus outside nodes that should be "seeded", including how you will handle skipping dataset edge evaluation when appropriate
- How you will allow optionally caching intermediate graph payloads for auxiliary data passing between graph loss terms
- What the format of the cache will look like and how graph loss terms will read it and append their intermediate results to it
- What options are available for limiting/controlling the size of the cache, e.g. detecting which terms need which parts of the cache, and then cleaning those parts of the cache that aren't needed anymore. Please include a detailed plan for how you will manage optimal ordering of terms to limit the size of the cache.
- How you will implement and pass unit tests for all new features

## Testing and done-ness criteria
At a minimum, you should include tests that cover:

- Auto-generation of graph loss terms to be used in GraphLoss and Train, and configuration from yaml
- Auto-generation for a small 4-node cyclic graph, checking that all 16 terms are correctly generated and that they can all be evaluated as expected using the error ops belonging to the destination nodes of all cycle path errors
- Check that "outside" node seeding works for the 4-node cyclic graph, e.g. the shortest path is taken from the dataset nodes to the outside nodes
- Intermediate graph path payloads can be cached and reused via auxiliary data passing between generated graph terms for the 4-node cyclic graph
- The size of the cache can be controlled and the terms are optimally ordered to limit the number of cached terms
- Cache items are deleted when they are no longer needed

You should implement these tests in a new test_loss.py file. Also write any other unit tests for new features. Iterate until all affected tests pass.