/*
 * cbgl - [IROS'24] Globally localise your 2D LIDAR in a 2D map in no time
 *
 * Copyright (c) 2024 Alexandros PHILOTHEOU
 *
 * Licensed under the MIT License.
 * See LICENSE.MIT for details.
 */
#include <cbgl_node/cbgl.h>
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("cbgl_node");
  // ros::NodeHandle nh;
  // ros::NodeHandle nh_private("~");

  CBGL cbgl(nh, nh_private);
  rclcpp::spin();
  rclcpp::shutdown();
  return 0;
}
