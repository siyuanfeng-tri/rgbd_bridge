#include "rgbd_bridge/real_sense_sr300.h"

#include <boost/make_shared.hpp>

namespace rgbd_bridge {
namespace {

void print_extrinsics(const rs::extrinsics& ex) {
  std::cout << ex.rotation[0] << ", " << ex.rotation[1] << ", " << ex.rotation[2] << ", " << ex.translation[0] << "\n";
  std::cout << ex.rotation[3] << ", " << ex.rotation[4] << ", " << ex.rotation[5] << ", " << ex.translation[1] << "\n";
  std::cout << ex.rotation[6] << ", " << ex.rotation[7] << ", " << ex.rotation[8] << ", " << ex.translation[2] << "\n";
}

void rs_extrinsics_to_eigen(const rs::extrinsics& ex, Eigen::Isometry3f* tf) {
  tf->setIdentity();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tf->linear()(i, j) = ex.rotation[3 * i + j];
    }
    tf->translation()(i) = ex.translation[i];
  }
}

void eigen_to_rs_extrinsics(const Eigen::Isometry3f& tf, rs::extrinsics* ex) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ex->rotation[3 * i + j] = tf.linear()(i, j);
    }
    ex->translation[i] = tf.translation()(i);
  }
}

boost::shared_ptr<cv::Mat> MakeImg(const void *src, int width, int height,
                                   rs::format pixel_type) {
  int cv_type = -1;
  switch (pixel_type) {
  case rs::format::z16:
  case rs::format::disparity16:
  case rs::format::y16:
  case rs::format::raw16:
    cv_type = CV_16UC1;
    break;
  case rs::format::y8:
  case rs::format::raw8:
    cv_type = CV_8UC1;
    break;
  case rs::format::rgb8:
  case rs::format::bgr8:
    cv_type = CV_8UC3;
    break;
  case rs::format::rgba8:
  case rs::format::bgra8:
    cv_type = CV_8UC4;
    break;
  default:
    throw std::runtime_error("Doesn't support this type");
  }

  boost::shared_ptr<cv::Mat> img =
      boost::make_shared<cv::Mat>(height, width, cv_type);
  if (cv_type == CV_8UC3) {
    memcpy(img->data, src, sizeof(uint8_t) * 3 * width * height);
  } else if (cv_type == CV_8UC4) {
    memcpy(img->data, src, sizeof(uint8_t) * 4 * width * height);
  } else if (cv_type == CV_8UC1) {
    memcpy(img->data, src, sizeof(uint8_t) * width * height);
  } else if (cv_type == CV_16UC1) {
    memcpy(img->data, src, sizeof(uint16_t) * width * height);
  } else {
    throw std::runtime_error("Doesn't support this type");
  }

  return img;
}

template <typename Scalar> void ScaleImg(cv::Mat *img, float scale) {
  for (int r = 0; r < img->rows; r++) {
    for (int c = 0; c < img->cols; c++) {
      img->at<Scalar>(r, c) *= scale;
    }
  }
}

void ScaleImgInPlace(cv::Mat *img, float scale) {
  switch (img->type()) {
  case CV_8UC4:
    ScaleImg<cv::Vec4b>(img, scale);
    break;
  case CV_8UC3:
    ScaleImg<cv::Vec3b>(img, scale);
    break;
  case CV_8UC1:
    ScaleImg<uint8_t>(img, scale);
    break;
  case CV_16UC1:
    ScaleImg<uint16_t>(img, scale);
    break;
  default:
    throw std::runtime_error("Doesn't support this type");
  }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
MakePointCloud(const RealSenseSR300 &camera, const rs::intrinsics &depth_intrin,
               const rs::intrinsics &color_intrin,
               const rs::extrinsics &depth_to_color,
               const rs::extrinsics &depth_to_desired, float depth_scale,
               const uint8_t *color_image, const uint16_t *depth_image) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  for (int dy = 0; dy < depth_intrin.height; ++dy) {
    for (int dx = 0; dx < depth_intrin.width; ++dx) {
      // Retrieve the 16-bit depth value and map it into a depth in meters
      uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
      float depth_in_meters = depth_value * depth_scale;

      // Skip over pixels with a depth value of zero, which is used to
      // indicate no data
      if (depth_value == 0 || depth_in_meters > 1 || depth_in_meters < 0.11)
        continue;

      // Map from pixel coordinates in the depth image to pixel coordinates in
      // the color image
      rs::float2 depth_pixel = {(float)dx, (float)dy};
      rs::float3 depth_point =
          depth_intrin.deproject(depth_pixel, depth_in_meters);

      rs::float3 color_point = depth_to_color.transform(depth_point);
      rs::float2 color_pixel = color_intrin.project(color_point);
      rs::float3 desired_point = depth_to_desired.transform(depth_point);

      // Use the color from the nearest color pixel, or pure white if this
      // point falls outside the color image
      const int cx = (int)std::round(color_pixel.x),
                cy = (int)std::round(color_pixel.y);
      pcl::PointXYZRGB point;
      point.x = desired_point.x;
      point.y = desired_point.y;
      point.z = desired_point.z;
      if (cx < 0 || cy < 0 || cx >= color_intrin.width ||
          cy >= color_intrin.height) {
        point.r = 255;
        point.g = 165;
        point.b = 0;
      } else {
        point.r = color_image[(cy * color_intrin.width + cx) * 3];
        point.g = color_image[(cy * color_intrin.width + cx) * 3 + 1];
        point.b = color_image[(cy * color_intrin.width + cx) * 3 + 2];
      }
      cloud->push_back(point);
    }
  }
  return cloud;
}
}

rs::context RealSenseSR300::context_;

RealSenseSR300::RealSenseSR300(int camera_id)
    : RGBDSensor({ImageType::RGB, ImageType::DEPTH, ImageType::IR,
                  ImageType::RECT_RGB, ImageType::RECT_RGB_ALIGNED_DEPTH,
                  ImageType::DEPTH_ALIGNED_RGB}) {
  const int num_cameras = context_.get_device_count();
  if (num_cameras <= camera_id) {
    throw std::runtime_error("Device " + std::to_string(camera_id) +
                             " not detected. Is it plugged in?");
  }
  camera_ = context_.get_device(camera_id);

  // Get supported ImageType.
  const auto &supported_types = get_supported_image_types();
  camera_->enable_stream(rs::stream::color, rs::preset::best_quality);
  camera_->enable_stream(rs::stream::depth, rs::preset::best_quality);
  camera_->enable_stream(rs::stream::infrared, rs::preset::best_quality);
  for (const auto &type : supported_types) {
    rs::stream stream = ImageTypeToStreamType(type);
    intrinsics_[stream] = camera_->get_stream_intrinsics(stream);
  }

  // Get default extrinsics.
  rs::extrinsics ex = camera_->get_extrinsics(rs::stream::depth, rs::stream::color);
  rs_extrinsics_to_eigen(ex, &ir_to_rgb_);
}

bool RealSenseSR300::IsObjectInGrasp(const cv::Mat &raw_depth,
                                     double depth_thresh_m) const {
  // This is 7cm.
  const uint16_t depth_thresh = (uint16_t)(depth_thresh_m * 1000);
  for (int h = 479; h >= 450; h--) {
    int ctr = 0;
    for (int w = 46; w < 320; w++) {
      uint16_t pixel = raw_depth.at<uint16_t>(h, w);
      if (pixel < depth_thresh && pixel != 0) {
        ctr++;
      } else {
        ctr = 0;
      }
    }
    // Has at least 5 continuous pixel that are not empty and
    // closer than depth thresh.
    if (ctr < 5)
      return false;
  }
  return true;
  // std::cout << "Close pixel count: " << ctr << "\n";
}

void RealSenseSR300::set_laser_projector_power(int level) {
  // sfeng: for some wtf reaseon, i need to set this in a tight loop to get it
  // to work.
  for (int i = 0; i < 3; i++) {
    camera_->set_option(rs::option::f200_laser_power, level);
  }
  assert(camera_->get_option(rs::option::f200_laser_power) == level);
}

Eigen::Vector2f RealSenseSR300::Project(const ImageType type,
                                        const Eigen::Vector3f &xyz) const {
  const rs::stream stream = ImageTypeToStreamType(type);
  const rs::intrinsics &intrinsics = intrinsics_.at(stream);
  rs::float3 xyz1 = {xyz(0), xyz(1), xyz(2)};
  rs::float2 pixel = intrinsics.project(xyz1);
  return Eigen::Vector2f(pixel.x, pixel.y);
}

Eigen::Vector3f RealSenseSR300::Deproject(const ImageType type,
                                          const Eigen::Vector2i &uv,
                                          float depth) const {
  const rs::stream stream = ImageTypeToStreamType(type);
  const rs::intrinsics &intrinsics = intrinsics_.at(stream);
  rs::float2 pixel = {(float)uv(0), (float)uv(1)};
  rs::float3 xyz = intrinsics.deproject(pixel, depth);
  return Eigen::Vector3f(xyz.x, xyz.y, xyz.z);
}

int RealSenseSR300::get_number_of_cameras() {
  return context_.get_device_count();
}

void RealSenseSR300::Start(const std::vector<ImageType> &types,
                           const ImageType cloud_base) {
  if (run_) {
    return;
  }

  run_ = true;
  std::cout << std::string(camera_->get_name()) << "@"
            << std::string(camera_->get_usb_port_id()) << " starting.\n";

  /////////////////////////////////////////////////////////////////////////////
  // Config params.
  set_mode(rs_ivcam_preset::RS_IVCAM_PRESET_DEFAULT);
  cloud_base_ = ImageTypeToStreamType(cloud_base);

  /////////////////////////////////////////////////////////////////////////////
  // Reset shared data.
  for (const ImageType type : types) {
    images_[type] = TimeStampedImage();
    std::cout << ImageTypeToString(type) << "\n";
    auto intr = intrinsics_.at(ImageTypeToStreamType(type));
    std::cout << "fx: " << intr.fx << "\n";
    std::cout << "fy: " << intr.fy << "\n";
    std::cout << "ppx: " << intr.ppx << "\n";
    std::cout << "ppy: " << intr.ppy << "\n";
    std::cout << "coeffs[0]: " << intr.coeffs[0] << "\n";
    std::cout << "coeffs[1]: " << intr.coeffs[1] << "\n";
    std::cout << "coeffs[2]: " << intr.coeffs[2] << "\n";
    std::cout << "coeffs[3]: " << intr.coeffs[3] << "\n";
    std::cout << "coeffs[4]: " << intr.coeffs[4] << "\n";
  }

  /////////////////////////////////////////////////////////////////////////////
  // Start thread.
  camera_->start();

  thread_ = std::thread(&RealSenseSR300::PollingThread, this);
}

void RealSenseSR300::Stop() {
  run_ = false;
  thread_.join();
  camera_->stop();
  // Clean up.
  images_.clear();
  cloud_ = TimeStampedCloud();
}

void RealSenseSR300::set_mode(const rs_ivcam_preset mode) {
  apply_ivcam_preset(camera_, mode);
}

rs::stream RealSenseSR300::ImageTypeToStreamType(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return rs::stream::color;
  case ImageType::DEPTH:
    return rs::stream::depth;
  case ImageType::IR:
    return rs::stream::infrared;
  case ImageType::RECT_RGB:
    return rs::stream::rectified_color;
  case ImageType::RECT_RGB_ALIGNED_DEPTH:
    return rs::stream::depth_aligned_to_rectified_color;
  case ImageType::DEPTH_ALIGNED_RGB:
    return rs::stream::color_aligned_to_depth;
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

bool RealSenseSR300::is_enabled(const ImageType type) const {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  return camera_->is_stream_enabled(ImageTypeToStreamType(type));
}

float RealSenseSR300::get_focal_length_x(const ImageType type) const {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  return intrinsics_.at(ImageTypeToStreamType(type)).fx;
}

float RealSenseSR300::get_focal_length_y(const ImageType type) const {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  return intrinsics_.at(ImageTypeToStreamType(type)).fy;
}

float RealSenseSR300::get_principal_point_x(const ImageType type) const {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  return intrinsics_.at(ImageTypeToStreamType(type)).ppx;
}

float RealSenseSR300::get_principal_point_y(const ImageType type) const {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  return intrinsics_.at(ImageTypeToStreamType(type)).ppy;
}

void RealSenseSR300::set_focal_length_x(const ImageType type, float val) {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  intrinsics_.at(ImageTypeToStreamType(type)).fx = val;
  assert(get_focal_length_x(type) == val);
}

void RealSenseSR300::set_focal_length_y(const ImageType type, float val) {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  intrinsics_.at(ImageTypeToStreamType(type)).fy = val;
  assert(get_focal_length_y(type) == val);
}

void RealSenseSR300::set_principal_point_x(const ImageType type, float val) {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  intrinsics_.at(ImageTypeToStreamType(type)).ppx = val;
  assert(get_principal_point_x(type) == val);
}

void RealSenseSR300::set_principal_point_y(const ImageType type, float val) {
  if (!camera_)
    throw std::runtime_error("No Camera available.");
  intrinsics_.at(ImageTypeToStreamType(type)).ppy = val;
  assert(get_principal_point_y(type) == val);
}

void RealSenseSR300::PollingThread() {
  rs::device &camera = *camera_;

  rs::intrinsics depth_intrin = intrinsics_.at(rs::stream::depth);
  rs::intrinsics color_intrin = intrinsics_.at(rs::stream::color);
  rs::extrinsics depth_to_color, depth_to_desired;
  eigen_to_rs_extrinsics(ir_to_rgb_, &depth_to_color);
  if (cloud_base_ == rs::stream::color ||
      cloud_base_ == rs::stream::rectified_color) {
    eigen_to_rs_extrinsics(ir_to_rgb_, &depth_to_desired);
  } else {
    depth_to_desired = camera.get_extrinsics(rs::stream::depth, cloud_base_);
  }

  float depth_scale = camera.get_depth_scale();

  // This should contain all the enabled streams.
  std::map<const ImageType, TimeStampedImage> images = images_;

  while (run_) {
    // Block until all frames have arrived.
    camera.wait_for_frames();

    // TODO(Jiaji): Use the points directly to reconstruct the point cloud
    // as in cpp_pointcloud.

    // Raw color and depth img.
    const uint8_t *color_image =
        (const uint8_t *)camera.get_frame_data(rs::stream::color);

    const uint16_t *depth_image =
        (const uint16_t *)camera.get_frame_data(rs::stream::depth);

    // Make point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
        MakePointCloud(*this, depth_intrin, color_intrin, depth_to_color,
                       depth_to_desired, depth_scale, color_image, depth_image);

    // Make cv::Mat of the images.
    // Jiaji: Trying out depth_aligned_to_rect_rgb.
    for (auto &pair : images) {
      rs::stream stream = ImageTypeToStreamType(pair.first);
      pair.second.timestamp = camera.get_frame_timestamp(stream);
      pair.second.count = camera.get_frame_number(stream);
      boost::shared_ptr<cv::Mat> img = MakeImg(
          camera.get_frame_data(stream), camera.get_stream_width(stream),
          camera.get_stream_height(stream), camera.get_stream_format(stream));
      // Scale depth image to units of mm.
      if (is_depth_image(pair.first)) {
        ScaleImgInPlace(img.get(), depth_scale * 1e3);
      }

      pair.second.data = img;
    }

    // Lock and do pointer assignment.
    std::unique_lock<std::mutex> lock2(lock_);
    images_ = images;

    cloud_.timestamp = camera.get_frame_timestamp(rs::stream::depth);
    cloud_.count = camera.get_frame_number(rs::stream::depth);
    cloud_.data = cloud;
  }
}

boost::shared_ptr<const cv::Mat>
RealSenseSR300::DoGetLatestImage(const ImageType type,
                                 uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(lock_);
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

pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
RealSenseSR300::DoGetLatestPointCloud(uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(lock_);
  if (cloud_.count == 0) {
    *timestamp = 0;
    return nullptr;
  }
  *timestamp = cloud_.timestamp;
  return cloud_.data;
}

void RealSenseSR300::set_ir_to_rgb_extrinsics(const Eigen::Isometry3f& tf) {
  ir_to_rgb_ = tf;
}

} // namespace rgbd_bridge
