"""Enable ``python -m vibecheck ...`` as an alias for the ``vibecheck`` console
script. Handy when the script directory is not on PATH (for example a virtualenv
you have not activated): ``python -m vibecheck --version``,
``python -m vibecheck verify <query.vnnlib> --network N=<model.onnx>``, etc.
"""
import sys

from .pipeline import main

sys.exit(main())
