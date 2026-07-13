FROM texlive/texlive:latest-small@sha256:e36cbf0b9561c4c975858410abe815c6df7a73e5b7b687d36cdd4a1606164cee

# Packages required by the unmodified AAAI-27 style and this manuscript.
# The base image digest and the derived image ID are both recorded by the
# fail-closed paper build manifest.
RUN tlmgr install \
    booktabs \
    caption \
    courier \
    fontaxes \
    helvetic \
    mweights \
    natbib \
    newtx \
    placeins \
    psnfss

WORKDIR /work
