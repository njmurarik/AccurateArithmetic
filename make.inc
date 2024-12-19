#####################################################
#   CompResearch Make Include File    				#
#####################################################
# ---------------------------------------------------------------------
# Compiler selection 
# --------------------------------------------------------------------- 
CC = gcc
# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------
COPT = -O3 -march=native -mtune=native
# ---------------------------------------------------------------------
# Include directory
# ---------------------------------------------------------------------
CINC = -I/usr/local/Cellar/gmp/6.3.0/include/ -I/usr/local/Cellar/mpfr/4.2.1/include/ -I/usr/local/Cellar/libmpc/1.3.1/include/ -I/usr/local/opt/lapack/include/ -I/usr/local/opt/openblas/include/
# ---------------------------------------------------------------------
# Library directory
# ---------------------------------------------------------------------
CLIB = -L/usr/local/Cellar/gmp/6.3.0/lib/ -L/usr/local/Cellar/mpfr/4.2.1/lib/ -L/usr/local/Cellar/libmpc/1.3.1/lib/ -L/usr/local/opt/lapack/lib/ -L/usr/local/opt/openblas/lib/
# ---------------------------------------------------------------------
# Link and include flags
# ---------------------------------------------------------------------
CFLAGS = -lm -lmpfr -lgmp -lmpc -llapacke -llapack -lcblas