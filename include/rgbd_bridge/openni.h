#include "rgbd_bridge/rgbd_bridge.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/openni2_grabber.h>

#include <mutex>

namespace rgbd_bridge {

// This class wraps over the openni grabber class to provide services to access
// the point cloud.
class OpenNiComm : public RGBDSensor {
public:
  OpenNiComm();
  ~OpenNiComm() {
    Stop();
    cloud_connection_.disconnect();
    image_connection_.disconnect();
  }

  void Start(const std::vector<ImageType> &types,
             const ImageType cloud_camera) override;
  void Stop() override;

  bool is_enabled(const ImageType type) const override;

  Eigen::Vector2f Project(const ImageType type,
                          const Eigen::Vector3f &xyz) const override;
  // depth is in m. uv is in pixel.
  Eigen::Vector3f Deproject(const ImageType type, const Eigen::Vector2i &uv,
                            float depth) const override;

  float get_focal_length_x(const ImageType type) const override;
  float get_focal_length_y(const ImageType type) const override;
  float get_principal_point_x(const ImageType type) const override;
  float get_principal_point_y(const ImageType type) const override;

private:
  boost::shared_ptr<const cv::Mat>
  DoGetLatestImage(const ImageType type, uint64_t *timestamp) const override;

  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
  DoGetLatestPointCloud(uint64_t *timestamp) const override;

  void CloudCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void ImageCallback(const pcl::io::openni2::Image::Ptr &image,
                     const pcl::io::openni2::DepthImage::Ptr &depth,
                     float reciprocal_focal_length);

  pcl::io::OpenNI2Grabber camera_interface_;

  mutable std::mutex cloud_mutex_;
  mutable std::mutex image_mutex_;

  TimeStampedCloud cloud_{};
  std::map<const ImageType, TimeStampedImage> images_;

  bool is_running_{false};

  boost::signals2::connection cloud_connection_;
  boost::signals2::connection image_connection_;
};

} // namespace rgbd_bridge
