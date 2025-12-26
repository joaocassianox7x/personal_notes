html:
	sphinx-build -b html docs docs/_build/

run_html:
	sphinx-autobuild -b html docs docs/_build/

clean:
	rm -rf docs/_build/*
