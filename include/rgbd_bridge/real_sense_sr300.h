#pragma once

#include "rgbd_bridge/rgbd_bridge.h"
#include <atomic>
#include <librealsense2/rs.hpp>
#include <memory>
#include <mutex>
#include <thread>

namespace rgbd_bridge {

/**
 *
 */
class RealSenseSR300 : public RGBDSensor {
public:
  RealSenseSR300(int camera_id = 0);

  void Start(const std::vector<ImageType> &types,
             const ImageType cloud_camera) override;
  void Stop() override;

  // void set_mode(const rs_ivcam_preset mode);

  bool is_enabled(const ImageType type) const override;

  Eigen::Vector2f Project(const ImageType type,
                          const Eigen::Vector3f &xyz) const override;
  // depth is in m. uv is in pixel.
  Eigen::Vector3f Deproject(const ImageType type, const Eigen::Vector2i &uv,
                            float depth) const override;

  /*
  void set_focal_length_x(const ImageType type, float val) override;
  void set_focal_length_y(const ImageType type, float val) override;
  void set_principal_point_x(const ImageType type, float val) override;
  void set_principal_point_y(const ImageType type, float val) override;

  float get_focal_length_x(const ImageType type) const override;
  float get_focal_length_y(const ImageType type) const override;
  float get_principal_point_x(const ImageType type) const override;
  float get_principal_point_y(const ImageType type) const override;
  */

  Intrinsics get_intrinsics(const ImageType type) const override;

  static int get_number_of_cameras();

  void set_laser_projector_power(int level);

  bool IsObjectInGrasp(const cv::Mat &raw_depth, double depth_thresh_m) const;

private:
  boost::shared_ptr<const cv::Mat>
  DoGetLatestImage(const ImageType type, uint64_t *timestamp) const override;

  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
  DoGetLatestPointCloud(uint64_t *timestamp) const override;

  rs2_stream ImageTypeToStreamType(const ImageType type) const;

  void PollingThread();

  rs2::device camera_;
  rs2::pipeline pipeline_;
  rs2::config config_;
  static rs2::context context_;

  std::map<rs2_stream, rs2::stream_profile> stream_profiles_;
  std::map<rs2_stream, rs2_intrinsics> intrinsics_;
  float depth_scale_;

  std::atomic<bool> run_{false};
  mutable std::mutex lock_;
  std::thread thread_;
  std::map<const ImageType, TimeStampedImage> images_;
  TimeStampedCloud cloud_{};
  rs2_stream cloud_base_;
};

} // namespace rgbd_bridge
