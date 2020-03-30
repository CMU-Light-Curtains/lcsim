cmake_minimum_required(VERSION 2.8.3)
project(lcsim)

set (CMAKE_CXX_STANDARD 11)

find_package(PythonLibs 3)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# Alglib
file(GLOB alglib_SRC
		"alglib/src/*.cpp"
		"alglib/include/*.h"
)

add_library(alglib ${alglib_SRC})
set_property(TARGET alglib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Lcsim
file(GLOB lcsim_SRC
    "lcsim/include/lcsim/*.h"
    "lcsim/src/*.cpp"
)
message("${lcsim_SRC}")

include_directories(
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	lcsim/include/lcsim
	alglib/include/
	python
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(lcsim ${lcsim_SRC})
target_link_libraries(
  lcsim
  alglib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
set_target_properties(lcsim PROPERTIES LINKER_LANGUAGE CXX)

# ROS Flag
if (${ROS})
	add_definitions(-DROS)
	message(STATUS "Building with ROS support.")
	find_package(catkin REQUIRED COMPONENTS
			pcl_conversions
			pcl_ros
			roscpp
			rospy
			cv_bridge
			geometry_msgs
			std_msgs
	)
	include_directories(${catkin_INCLUDE_DIRS})
else()
	set(catkin_LIBRARIES "")
endif()

# Pylc
add_subdirectory(pybind11)
pybind11_add_module(pylc_lib python/pylc_lib.cpp)
target_link_libraries(pylc_lib PRIVATE pybind11::module alglib lcsim ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${catkin_LIBRARIES})
