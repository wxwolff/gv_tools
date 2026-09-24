Notebooks
=========

Release notebooks are kept in the project-level ``notebooks`` folder:

* ``GV_Tools_PIERS0042_20260810.ipynb``
* ``GV_Tools_Ingest_Demonstration.ipynb`` (canonical master notebook)
* ``MetOne_AIO_GAIL_sample.ipynb``
* ``RM_Young_AIO_PIERS_sample.ipynb``

The installed location can be discovered without hard-coded paths:

.. code-block:: python

   from pathlib import Path

   notebooks = Path("notebooks")

The notebooks may be launched from any directory. They use
``GV_TOOLS_HOME`` when set and otherwise default to
``~/Desktop/Work/GV Tools``. Input and output locations can also be overridden
with the environment variables documented in each configuration cell.
