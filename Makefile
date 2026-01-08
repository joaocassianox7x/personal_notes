html:
	rm -rf docs/_build/*
	sphinx-build -b html docs docs/_build/

run_html:
	rm -rf docs/_build/*
	sphinx-autobuild -b html docs docs/_build/

clean:
	rm -rf docs/_build/*
