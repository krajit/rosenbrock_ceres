# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ajit/Desktop/rosenbrock_ceres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ajit/Desktop/rosenbrock_ceres/build

# Include any dependencies generated for this target.
include CMakeFiles/rosenbrock.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rosenbrock.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rosenbrock.dir/flags.make

CMakeFiles/rosenbrock.dir/rosenbrock.cc.o: CMakeFiles/rosenbrock.dir/flags.make
CMakeFiles/rosenbrock.dir/rosenbrock.cc.o: ../rosenbrock.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ajit/Desktop/rosenbrock_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rosenbrock.dir/rosenbrock.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rosenbrock.dir/rosenbrock.cc.o -c /home/ajit/Desktop/rosenbrock_ceres/rosenbrock.cc

CMakeFiles/rosenbrock.dir/rosenbrock.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rosenbrock.dir/rosenbrock.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ajit/Desktop/rosenbrock_ceres/rosenbrock.cc > CMakeFiles/rosenbrock.dir/rosenbrock.cc.i

CMakeFiles/rosenbrock.dir/rosenbrock.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rosenbrock.dir/rosenbrock.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ajit/Desktop/rosenbrock_ceres/rosenbrock.cc -o CMakeFiles/rosenbrock.dir/rosenbrock.cc.s

CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.requires:

.PHONY : CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.requires

CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.provides: CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.requires
	$(MAKE) -f CMakeFiles/rosenbrock.dir/build.make CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.provides.build
.PHONY : CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.provides

CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.provides.build: CMakeFiles/rosenbrock.dir/rosenbrock.cc.o


# Object files for target rosenbrock
rosenbrock_OBJECTS = \
"CMakeFiles/rosenbrock.dir/rosenbrock.cc.o"

# External object files for target rosenbrock
rosenbrock_EXTERNAL_OBJECTS =

rosenbrock: CMakeFiles/rosenbrock.dir/rosenbrock.cc.o
rosenbrock: CMakeFiles/rosenbrock.dir/build.make
rosenbrock: /home/ajit/ceres-bin/lib/libceres.a
rosenbrock: /usr/local/lib/libglog.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
rosenbrock: /usr/lib/x86_64-linux-gnu/libspqr.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libcholmod.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libccolamd.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libcamd.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libcolamd.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libamd.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libopenblas.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
rosenbrock: /usr/lib/x86_64-linux-gnu/librt.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libcxsparse.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libopenblas.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
rosenbrock: /usr/lib/x86_64-linux-gnu/librt.so
rosenbrock: /usr/lib/x86_64-linux-gnu/libcxsparse.so
rosenbrock: CMakeFiles/rosenbrock.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ajit/Desktop/rosenbrock_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rosenbrock"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rosenbrock.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rosenbrock.dir/build: rosenbrock

.PHONY : CMakeFiles/rosenbrock.dir/build

CMakeFiles/rosenbrock.dir/requires: CMakeFiles/rosenbrock.dir/rosenbrock.cc.o.requires

.PHONY : CMakeFiles/rosenbrock.dir/requires

CMakeFiles/rosenbrock.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rosenbrock.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rosenbrock.dir/clean

CMakeFiles/rosenbrock.dir/depend:
	cd /home/ajit/Desktop/rosenbrock_ceres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ajit/Desktop/rosenbrock_ceres /home/ajit/Desktop/rosenbrock_ceres /home/ajit/Desktop/rosenbrock_ceres/build /home/ajit/Desktop/rosenbrock_ceres/build /home/ajit/Desktop/rosenbrock_ceres/build/CMakeFiles/rosenbrock.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rosenbrock.dir/depend

