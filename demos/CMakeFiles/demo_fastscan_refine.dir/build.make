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
include demos/CMakeFiles/demo_fastscan_refine.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demos/CMakeFiles/demo_fastscan_refine.dir/compiler_depend.make

# Include the progress variables for this target.
include demos/CMakeFiles/demo_fastscan_refine.dir/progress.make

# Include the compile flags for this target's objects.
include demos/CMakeFiles/demo_fastscan_refine.dir/flags.make

demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o: demos/CMakeFiles/demo_fastscan_refine.dir/flags.make
demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o: demos/demo_fastscan_refine.cpp
demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o: demos/CMakeFiles/demo_fastscan_refine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhan4404/disk-based/faiss4_disk_vector/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o -MF CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o.d -o CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o -c /home/zhan4404/disk-based/faiss4_disk_vector/demos/demo_fastscan_refine.cpp

demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.i"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhan4404/disk-based/faiss4_disk_vector/demos/demo_fastscan_refine.cpp > CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.i

demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.s"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhan4404/disk-based/faiss4_disk_vector/demos/demo_fastscan_refine.cpp -o CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.s

# Object files for target demo_fastscan_refine
demo_fastscan_refine_OBJECTS = \
"CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o"

# External object files for target demo_fastscan_refine
demo_fastscan_refine_EXTERNAL_OBJECTS =

demos/demo_fastscan_refine: demos/CMakeFiles/demo_fastscan_refine.dir/demo_fastscan_refine.cpp.o
demos/demo_fastscan_refine: demos/CMakeFiles/demo_fastscan_refine.dir/build.make
demos/demo_fastscan_refine: faiss/libfaiss_avx2.a
demos/demo_fastscan_refine: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
demos/demo_fastscan_refine: /usr/lib/x86_64-linux-gnu/libmkl_sequential.so
demos/demo_fastscan_refine: /usr/lib/x86_64-linux-gnu/libmkl_core.so
demos/demo_fastscan_refine: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
demos/demo_fastscan_refine: /usr/lib/x86_64-linux-gnu/libpthread.a
demos/demo_fastscan_refine: demos/CMakeFiles/demo_fastscan_refine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhan4404/disk-based/faiss4_disk_vector/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_fastscan_refine"
	cd /home/zhan4404/disk-based/faiss4_disk_vector/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_fastscan_refine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/CMakeFiles/demo_fastscan_refine.dir/build: demos/demo_fastscan_refine
.PHONY : demos/CMakeFiles/demo_fastscan_refine.dir/build

demos/CMakeFiles/demo_fastscan_refine.dir/clean:
	cd /home/zhan4404/disk-based/faiss4_disk_vector/demos && $(CMAKE_COMMAND) -P CMakeFiles/demo_fastscan_refine.dir/cmake_clean.cmake
.PHONY : demos/CMakeFiles/demo_fastscan_refine.dir/clean

demos/CMakeFiles/demo_fastscan_refine.dir/depend:
	cd /home/zhan4404/disk-based/faiss4_disk_vector && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhan4404/disk-based/faiss4_disk_vector /home/zhan4404/disk-based/faiss4_disk_vector/demos /home/zhan4404/disk-based/faiss4_disk_vector /home/zhan4404/disk-based/faiss4_disk_vector/demos /home/zhan4404/disk-based/faiss4_disk_vector/demos/CMakeFiles/demo_fastscan_refine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/CMakeFiles/demo_fastscan_refine.dir/depend

