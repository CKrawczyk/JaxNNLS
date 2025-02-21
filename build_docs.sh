awk '{gsub(/```math/,"```{math}")}1' README.md > docs/source/README.md
cd docs
make clean
make html
cd ..
