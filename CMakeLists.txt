cmake_minimum_required(VERSION 3.16)
project(DE_Solver)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOURCES JacobiMethodSolver.cpp)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
ADD_COMPILE_OPTIONS(-Wall -Wextra)

add_executable(DE_Solver ${SOURCES})
