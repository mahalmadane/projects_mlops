"""Allow ``python -m mlstrat``, same as the ``mlstrat`` command."""

import sys

from mlstrat.cli import main

sys.exit(main())
