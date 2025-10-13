cmake_minimum_required(VERSION 3.14.4)
project(cbgl)

# C++ Standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()
add_compile_options(-Ofast)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

# csm for scan--to--map-scan matching when λ < 2π
find_package(PkgConfig)
#pkg_check_modules(csm REQUIRED csm)
# CSM lokal einbinden
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../csm EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../csm/include)


# FSM for scan--to--map-scan matching when λ = 2π
# FSM needs CGAL
find_package(CGAL REQUIRED)

# include_directories(include
#   ${catkin_INCLUDE_DIRS}
#   ${csm_INCLUDE_DIRS}
#   /usr/include/eigen3/
# )
# Includes
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${csm_LIBRARIES}
  /usr/include/eigen3/
)#  ${csm_INCLUDE_DIRS}


# Occupancy grid utils
add_library(cbgl_og_utils_lib
  src/utils/occupancy_grid_utils/coordinate_conversions.cpp
  src/utils/occupancy_grid_utils/ray_tracer.cpp
)

# Particle filter utils
add_library(cbgl_pf_utils_lib
  src/utils/pf/pf.c
  src/utils/pf/pf_kdtree.c
  src/utils/pf/pf_pdf.c
  src/utils/pf/pf_vector.c
  src/utils/pf/eig3.c
)

# Map utils from generation of hypotheses
add_library(cbgl_map_utils_lib
  src/utils/map/map.c
)

target_link_libraries(cbgl_og_utils_lib
  cbgl_pf_utils_lib
  cbgl_map_utils_lib
  csm
  ${PCL_LIBRARIES}
) #${csm_LIBRARIES}

ament_target_dependencies(cbgl_og_utils_lib
  rclcpp
  geometry_msgs
  nav_msgs
  pcl_ros
)

# cbgl
add_subdirectory(src/cbgl_node)


ament_package()
