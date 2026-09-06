# Propuesta doctoral. Fuente canónica: el .tex
# Uso: make -C docs

TEX = propuesta_doctoral_seleccion_multifidelidad_rl.tex

.PHONY: all clean

all: $(TEX:.tex=.pdf)

%.pdf: %.tex %.bib
	latexmk -pdf -interaction=nonstopmode $<

clean:
	latexmk -c $(TEX)
