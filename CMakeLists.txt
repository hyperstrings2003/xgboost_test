cmake_minimum_required(VERSION 3.3)
project(xgboost_test)

SET(CMAKE_C_COMPILER "/Win_1TB/code/test_code/gcc-4.9.2/bin/gcc")
SET(CMAKE_CXX_COMPILER "/Win_1TB/code/test_code/gcc-4.9.2/bin/g++")

include_directories("/Win_1TB/Ketchum_Code/xgboost/include/")
include_directories("/Win_1TB/Ketchum_Code/xgboost/rabit/include/")
include_directories("/Win_1TB/Ketchum_Code/xgboost/dmlc-core/include/")

#link_directories("/Win_1TB/Ketchum_Code/xgboost/lib/")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(xgboost_test ${SOURCE_FILES})

target_link_libraries(xgboost_test "/Win_1TB/Ketchum_Code/xgboost/lib/libxgboost.so")