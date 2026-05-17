# Minimal makefile for Sphinx documentation
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile clean serve-html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR) examples sg_execution_times.rst api/_autosummary api/_backrefs

# Spin up a local http server to view the rendered documentation.
# This is required for interactive examples to work.
serve-html:
	uv run python -m http.server 11000 --directory "$(BUILDDIR)"/html

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
