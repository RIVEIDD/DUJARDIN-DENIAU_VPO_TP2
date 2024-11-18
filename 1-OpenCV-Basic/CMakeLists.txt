cmake_minimum_required ( VERSION 2.8)
project ( ImageProc-TP2 )

# sources extension
set(SRCEXT "cpp")

file(
    GLOB_RECURSE
    source_files
    *.${SRCEXT}
)

# required packages
find_package ( OpenCV REQUIRED )

# adding the source files to the executable
add_executable(ImageProc-TP2 main.cpp)


# linking other libraries
target_link_libraries(ImageProc-TP2  ${OpenCV_LIBS})
