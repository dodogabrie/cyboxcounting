# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.
help:
	@echo "Available tasks:"
	@echo "compile    -> Build the Cython extension module."
	@echo "annotate   -> Create annotated HTML from the .pyx sources"
	@echo "all        -> Call compile, clean-soft"
	@echo "clean      -> Remove *.so *.c *.o *.html build core"
	@echo "clean-soft -> Remove *.so *.c *.o build"
	@echo "test       -> Run the test.py"

all: create-build compile soft-clean

create-build: 
	mkdir -p build
compile:
	python3 setup.py build_ext --build-lib build
annotate:
	cython -3 -a src/*.pyx
	@echo "Annotated HTML of the code"
test:
	python3 test.py

# Phony targets for cleanup and similar uses
.PHONY: clean soft-clean
clean:
	rm -rf *.so *.c *.o build __pycache__ core data
soft-clean:
	rm -rf *.c *html __pycache__ 


# Suffix rules
%.c : %.pyx
	cython $<
