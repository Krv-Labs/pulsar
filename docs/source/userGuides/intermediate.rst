.. _intermediate:

============
Intermediate
============

Tune sweeps to control model diversity.

- Increase PCA dimensions and seeds to widen the representation grid.
- Expand ``ball_mapper.epsilon`` values to probe locality.
- Use ``cosmic_graph.threshold: auto`` or a fixed threshold to trade stability and sparsity.

.. code-block:: yaml

   sweep:
     pca:
       dimensions: {values: [2, 3, 4]}
       seed: {values: [1, 13, 42]}
     ball_mapper:
       epsilon: {range: {min: 0.2, max: 1.2, steps: 5}}
   cosmic_graph:
     threshold: auto
