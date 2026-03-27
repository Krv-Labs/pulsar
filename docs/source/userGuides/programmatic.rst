.. _programmatic:

============
Programmatic
============

The Python API can be built without YAML by constructing ``ThemaRS`` and
feeding a dataframe.

.. code-block:: python

   from pulsar import ThemaRS
   import pandas as pd

   df = pd.read_csv("data.csv")
   model = ThemaRS({
       "run": {"name": "example"},
       "preprocessing": {
           "impute": {"age": {"method": "sample_normal", "seed": 42}},
           "drop_columns": []
       },
       "sweep": {
           "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
           "ball_mapper": {"epsilon": {"values": [0.5]}}
       },
       "cosmic_graph": {"threshold": "auto"},
       "output": {"n_reps": 3}
   })
   model.fit(data=df)
   matrix = model.weighted_adjacency
   threshold = model.resolved_threshold
