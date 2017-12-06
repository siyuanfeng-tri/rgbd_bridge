#include "rgbd_bridge/openni.h"
#include "util/util.h"

#include <boost/make_shared.hpp>

namespace rgbd_bridge {

Eigen::Vector2f OpenNiComm::Project(const ImageType type,
                                    const Eigen::Vector3f &xyz) const {
  throw std::runtime_error("unimplemented");
}

Eigen::Vector3f OpenNiComm::Deproject(const ImageType type,
                                      const Eigen::Vector2i &uv,
                                      float depth) const {
  throw std::runtime_error("unimplemented");
}

OpenNiComm::OpenNiComm() : RGBDSensor({ImageType::RGB, ImageType::DEPTH}) {
  {
    boost::function<void(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &)>
        cloud_cb = boost::bind(&OpenNiComm::CloudCallback, this, _1);
    cloud_connection_ = camera_interface_.registerCallback(cloud_cb);
  }

  {
    boost::function<void(const pcl::io::openni2::Image::Ptr &,
                         const pcl::io::openni2::DepthImage::Ptr &,
                         float reciprocal_focal_length)>
        image_cb = boost::bind(&OpenNiComm::ImageCallback, this, _1, _2, _3);
    image_connection_ = camera_interface_.registerCallback(image_cb);
  }

  camera_interface_.setDepthCameraIntrinsics(550, 550, 297.31, 228.32);
  camera_interface_.setRGBCameraIntrinsics(515.9, 515.4, 304.53, 227.2);
};

void OpenNiComm::Start(const std::vector<ImageType> &types,
                       const ImageType cloud_base) {
  if (is_running_)
    return;

  is_running_ = true;

  images_[ImageType::RGB] = TimeStampedImage();
  images_[ImageType::DEPTH] = TimeStampedImage();

  camera_interface_.start();
}

void OpenNiComm::Stop() {
  if (!is_running_)
    return;
  camera_interface_.stop();
  images_.clear();
  cloud_ = TimeStampedCloud();
}

boost::shared_ptr<const cv::Mat>
OpenNiComm::DoGetLatestImage(const ImageType type, uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(image_mutex_);

  auto it = images_.find(type);
  if (it == images_.end()) {
    throw std::runtime_error("ImageType not initialized.");
  }
  const TimeStampedImage &image = it->second;

  if (image.count == 0) {
    *timestamp = 0;
    return nullptr;
  }
  *timestamp = image.timestamp;
  return image.data;
}

bool OpenNiComm::is_enabled(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
  case ImageType::DEPTH:
    return true;
  default:
    return false;
  }
}

pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
OpenNiComm::DoGetLatestPointCloud(uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(cloud_mutex_);
  if (cloud_.count == 0) {
    *timestamp = 0;
    return nullptr;
  }
  *timestamp = cloud_.timestamp;
  return cloud_.data;
}

void OpenNiComm::CloudCallback(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {
  std::unique_lock<std::mutex> lock(cloud_mutex_);
  cloud_.data = cloud;
  // Use the current system clock..
  cloud_.timestamp = get_system_time() * 1e6;
  cloud_.count++;
}

void OpenNiComm::ImageCallback(const pcl::io::openni2::Image::Ptr &image,
                               const pcl::io::openni2::DepthImage::Ptr &depth,
                               float reciprocal_focal_length) {
  auto color_img = boost::make_shared<cv::Mat>(image->getHeight(),
                                               image->getWidth(), CV_8UC3);
  image->fillRGB(color_img->cols, color_img->rows, color_img->data,
                 color_img->step);

  auto depth_img = boost::make_shared<cv::Mat>(depth->getHeight(),
                                               depth->getWidth(), CV_16UC1);
  depth->fillDepthImageRaw(depth_img->cols, depth_img->rows,
                           (uint16_t *)depth_img->data);

  std::unique_lock<std::mutex> lock(image_mutex_);
  auto &rgb = images_[ImageType::RGB];
  rgb.count++;
  rgb.data = color_img;
  rgb.timestamp = get_system_time() * 1e6;

  auto &d = images_[ImageType::DEPTH];
  d.count++;
  d.data = depth_img;
  d.timestamp = get_system_time() * 1e6;
}

float OpenNiComm::get_focal_length_x(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return 550;
  case ImageType::DEPTH:
    return 515.9;
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

float OpenNiComm::get_focal_length_y(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return 550;
  case ImageType::DEPTH:
    return 515.4;
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

float OpenNiComm::get_principal_point_x(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return 297.31;
  case ImageType::DEPTH:
    return 304.53;
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

float OpenNiComm::get_principal_point_y(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return 228.32;
  case ImageType::DEPTH:
    return 227.2;
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

} // namespace rgbd_bridge
