# Thin shim so `make <target>` muscle memory maps to the justfile.
# The real task definitions live in ./justfile (run `just --list`).
# Requires `just` (brew install just).

.PHONY: check build test test-all parity lint bootstrap hooks check-rust clean-sf

check build test test-all parity lint bootstrap hooks check-rust clean-sf:
	@just $@
