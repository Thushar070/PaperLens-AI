"""PaperLens Phase 9 API application."""

# Reuse the already installed Python 3.11 base packages when this project
# package is run from its existing isolated environment. This never installs
# packages or accesses the network.
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()
