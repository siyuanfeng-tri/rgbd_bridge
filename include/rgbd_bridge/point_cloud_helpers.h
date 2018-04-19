#pragma once

// This file contains helper functions for some operations involved in
// building point clouds.  It's not intended to be part of the
// external library API.  Code appears in the header file for better inlining.

#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rgbd_bridge {

/**
 * Makes a colored point cloud from a pair of color and depth images by
 * iterating through the depth image and find its corresponding pixel in the
 * color image for color registration. The resulting point cloud will be in the
 * desired frame specified by @p depth_to_desired.
 *
 * @param ignore_depth_not_in_rgb If set to true, will ignore all the points
 * that are not visible from the RGB camera.
 * @param color_intrin Intrinsics for the color camera. It's size need to
 * match @p color_image's dimension.
 * @param depth_intrin Intrinsics for the depth camera. It's size need to match
 * @p depth_image's dimension.
 * @param depth_to_color X_rgb_depth from the depth camera to the color camera.
 * @param depth_to_desired X_desired_depth from the depth camera to a desired
 * frame.
 * @param depth_scale Scales the contents in @p depth_image to meters.
 * @param near_z All points closer than this will be discarded, in meters.
 * @param far_z All points further than this will be discarded, in meters.
 * @param color_image RGB image. Need to be in RGB order.
 * @param depth_image Depth image.
 * @param[out] cloud_out Output dense point cloud. Contains no invalid points.
 * @param[out] organized_cloud_out Organized point cloud. For points that are
 * discard or does not have a depth return, the coordinates are set to NaN.
 */
inline void DoMakePointCloud(
    bool ignore_depth_not_in_rgb,
    const Intrinsics& color_intrin,
    const Intrinsics& depth_intrin,
    const Eigen::Isometry3f& depth_to_color,
    const Eigen::Isometry3f& depth_to_desired,
    float depth_scale, float near_z, float far_z,
    const cv::Mat& color_image, const cv::Mat& depth_image,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr* cloud_out,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr* organized_cloud_out) {
  if (depth_intrin.width() != depth_image.cols ||
      depth_intrin.height() != depth_image.rows) {
    throw std::runtime_error("Depth image dimension mismatch");
  }
  if (color_intrin.width() != color_image.cols ||
      color_intrin.height() != color_image.rows) {
    throw std::runtime_error("Color image dimension mismatch");
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  pcl::PointXYZRGB default_pt{};
  default_pt.x = std::numeric_limits<float>::quiet_NaN();
  default_pt.y = std::numeric_limits<float>::quiet_NaN();
  default_pt.z = std::numeric_limits<float>::quiet_NaN();

  cloud->height = depth_intrin.height();
  cloud->width = depth_intrin.width();
  cloud->is_dense = false;
  cloud->points.resize(cloud->height * cloud->width, default_pt);

  // Builds a dense organized cloud first in the depth camera's frame.
  for (int v = 0; v < depth_intrin.height(); ++v) {
    for (int u = 0; u < depth_intrin.width(); ++u) {
      uint16_t depth_value = depth_image.at<uint16_t>(v, u);
      float depth_in_meters = static_cast<float>(depth_value) * depth_scale;

      // Skip over pixels with a depth value of zero, which is used to
      // indicate no data
      if (depth_value == 0 || depth_in_meters > far_z ||
          depth_in_meters < near_z) {
        continue;
      }

      // Map from pixel coordinates in the depth image to pixel coordinates in
      // the color image
      Eigen::Vector3f depth_point =
          depth_intrin.BackProject(Eigen::Vector2f(u, v), depth_in_meters);

      Eigen::Vector3f color_point = depth_to_color * depth_point;
      Eigen::Vector2f color_pixel = color_intrin.Project(color_point);

      // Use the color from the nearest color pixel, skip if point falls out
      // the color image.
      const int cx = static_cast<int>(std::round(color_pixel(0)));
      const int cy = static_cast<int>(std::round(color_pixel(1)));

      // Set point in the cloud.
      pcl::PointXYZRGB& point = cloud->points.at(v * cloud->width + u);
      if (cx < 0 || cy < 0 || cx >= color_intrin.width() ||
          cy >= color_intrin.height()) {
        if (ignore_depth_not_in_rgb) {
          continue;
        }
        point.r = 255;
        point.g = 165;
        point.b = 0;
      } else {
        auto rgb = color_image.at<cv::Vec3b>(cy, cx);
        point.r = rgb[0];
        point.g = rgb[1];
        point.b = rgb[2];
      }
      point.x = depth_point(0);
      point.y = depth_point(1);
      point.z = depth_point(2);
    }
  }

  // Transform the cloud into the desired frame. No need to set the intrinsics
  // as long as the consumer use the trivial intrinsics.
  pcl::transformPointCloud(*cloud, *cloud, depth_to_desired);

  // Save results.
  if (organized_cloud_out) {
    *organized_cloud_out = cloud;
  }

  if (cloud_out) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dense_cloud =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    std::vector<int> unused_indices;
    pcl::removeNaNFromPointCloud(*cloud, *dense_cloud, unused_indices);
    *cloud_out = dense_cloud;
  }
}

}  // namespace rgbd_bridge
