# doen on juwels booster
# better to setup a virtual environment for pyhon otherwise there can be additional problems // python -m venv .venv

version=6.5.5
path=/p/project1/ngff/li47/code/temp/

cd ${path}
mkdir LHAPDF
cd LHAPDF
wget https://lhapdf.hepforge.org/downloads/LHAPDF-${version}.tar.gz
tar -xzf LHAPDF-${version}.tar.gz
mkdir build
cd LHAPDF-${version}
./configure --prefix=${path}LHAPDF/build

make
make install

# lhapdf install NNPDF40_nnlo_as_01180