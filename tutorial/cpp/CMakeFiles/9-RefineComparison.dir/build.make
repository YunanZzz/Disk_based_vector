# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhan4404/disk-based/faiss4_disk_vector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhan4404/disk-based/faiss4_disk_vector

# Include any dependencies generated for this target.
include tutorial/cpp/CMakeFiles/9-RefineComparison.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tutorial/cpp/CMakeFiles/9-RefineComparison.dir/compiler_depend.make

# Include the progress variables for this target.
include tutorial/cpp/CMakeFiles/9-RefineComparison.dir/progress.make

# Include the compile flags for this target's objects.
include tutorial/cpp/CMakeFiles/9-RefineComparison.dir/flags.make

tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o: tutorial/cpp/CMakeFiles/9-RefineComparison.dir/flags.make
tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o: tutorial/cpp/9-RefineComparison.cpp
tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o: tutorial/cpp/CMakeFiles/9-RefineComparison.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhan4404/disk-based/faiss4_disk_vector/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o -MF CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o.d -o CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o -c /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp/9-RefineComparison.cpp

tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.i"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp/9-RefineComparison.cpp > CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.i

tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.s"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp/9-RefineComparison.cpp -o CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.s

# Object files for target 9-RefineComparison
9__RefineComparison_OBJECTS = \
"CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o"

# External object files for target 9-RefineComparison
9__RefineComparison_EXTERNAL_OBJECTS =

tutorial/cpp/9-RefineComparison: tutorial/cpp/CMakeFiles/9-RefineComparison.dir/9-RefineComparison.cpp.o
tutorial/cpp/9-RefineComparison: tutorial/cpp/CMakeFiles/9-RefineComparison.dir/build.make
tutorial/cpp/9-RefineComparison: faiss/libfaiss_avx2.a
tutorial/cpp/9-RefineComparison: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
tutorial/cpp/9-RefineComparison: /usr/lib/x86_64-linux-gnu/libpthread.a
tutorial/cpp/9-RefineComparison: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
tutorial/cpp/9-RefineComparison: /usr/lib/x86_64-linux-gnu/libmkl_sequential.so
tutorial/cpp/9-RefineComparison: /usr/lib/x86_64-linux-gnu/libmkl_core.so
tutorial/cpp/9-RefineComparison: tutorial/cpp/CMakeFiles/9-RefineComparison.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhan4404/disk-based/faiss4_disk_vector/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 9-RefineComparison"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/9-RefineComparison.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tutorial/cpp/CMakeFiles/9-RefineComparison.dir/build: tutorial/cpp/9-RefineComparison
.PHONY : tutorial/cpp/CMakeFiles/9-RefineComparison.dir/build

tutorial/cpp/CMakeFiles/9-RefineComparison.dir/clean:
	cd /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp && $(CMAKE_COMMAND) -P CMakeFiles/9-RefineComparison.dir/cmake_clean.cmake
.PHONY : tutorial/cpp/CMakeFiles/9-RefineComparison.dir/clean

tutorial/cpp/CMakeFiles/9-RefineComparison.dir/depend:
	cd /home/zhan4404/disk-based/faiss4_disk_vector && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhan4404/disk-based/faiss4_disk_vector /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp /home/zhan4404/disk-based/faiss4_disk_vector /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp /home/zhan4404/disk-based/faiss4_disk_vector/tutorial/cpp/CMakeFiles/9-RefineComparison.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tutorial/cpp/CMakeFiles/9-RefineComparison.dir/depend

