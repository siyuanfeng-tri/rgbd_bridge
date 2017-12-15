#include "rgbd_bridge/real_sense_sr300.h"

#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>

using rgbd_bridge::ImageType;

struct UserClickedPt {
  int x{-1};
  int y{-1};
  std::string window_name;
};

cv::Mat proc_depth_image(const rgbd_bridge::RGBDSensor &camera,
                         const rgbd_bridge::ImageType type,
                         const cv::Mat &raw_depth, const UserClickedPt &input) {
  cv::Mat ret;
  ret = raw_depth;

  raw_depth.convertTo(ret, CV_8UC1, 255. / 1000., 0);
  cv::cvtColor(ret, ret, CV_GRAY2BGR);

  if (input.x >= 0 && input.x < ret.cols && input.y >= 0 &&
      input.y < ret.rows) {
    cv::circle(ret, cv::Point(input.x, input.y), 5, cv::Scalar(0, 255, 0));
    float depth = raw_depth.at<uint16_t>(input.y, input.x) / 1000.;
    Eigen::Vector3f xyz =
        camera.Deproject(type, Eigen::Vector2i(input.x, input.y), depth);

    const std::string text = "xyz: " + std::to_string(xyz(0)) + "," +
                             std::to_string(xyz(1)) + "," +
                             std::to_string(xyz(2));
    cv::putText(ret, text, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 255, 0));
  }

  /*
  if (type == rgbd_bridge::ImageType::DEPTH) {
    auto& real_sense = dynamic_cast<const
  rgbd_bridge::RealSenseSR300&>(camera);
    std::cout << "Is In grasp: " << real_sense.IsObjectInGrasp(raw_depth, 0.07)
  << "\n";
  }
  */

  return ret;
}

cv::Mat proc_ir_image(const cv::Mat &raw_ir, const UserClickedPt &input) {
  cv::Mat ret;
  cv::cvtColor(raw_ir, ret, CV_GRAY2BGR);
  return ret;
}

cv::Mat proc_rgb_image(const cv::Mat &raw_rgb, const UserClickedPt &input) {
  cv::Mat ret;
  cv::cvtColor(raw_rgb, ret, CV_RGB2BGR);
  return ret;
}

void image_loop(const rgbd_bridge::RGBDSensor *driver, int camera_id,
                const std::vector<rgbd_bridge::ImageType> channels,
                const std::vector<UserClickedPt> *clicked_xy) {
  uint64_t timestamp;

  cv::Mat tmp;
  while (true) {
    for (size_t i = 0; i < channels.size(); i++) {
      const auto type = channels[i];
      auto img = driver->GetLatestImage(type, &timestamp);
      if (img) {
        if (is_color_image(type)) {
          tmp = proc_rgb_image(*img, clicked_xy->at(i));
        } else if (is_depth_image(type)) {
          tmp = proc_depth_image(*driver, type, *img, clicked_xy->at(i));
        } else if (type == rgbd_bridge::ImageType::IR) {
          tmp = proc_ir_image(*img, clicked_xy->at(i));
        }

        cv::imshow(ImageTypeToString(type) + std::to_string(camera_id), tmp);
      }
    }

    cv::waitKey(5);
  }
}

void cloud_loop(const rgbd_bridge::RGBDSensor *driver) {
  uint64_t timestamp;

  pcl::visualization::PCLVisualizer viewer("Point Cloud Visualization");
  viewer.addCoordinateSystem(0.2, Eigen::Affine3f::Identity());

  while (true) {
    auto cloud = driver->GetLatestPointCloud(&timestamp);
    if (cloud) {
      viewer.addPointCloud(cloud, "cloud");
      viewer.spinOnce();
      viewer.removePointCloud("cloud");
    }
  }
}

void mouse_click(int event, int x, int y, int flags, void *userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    UserClickedPt *data = (UserClickedPt *)userdata;
    data->x = x;
    data->y = y;
    std::cout << data->window_name << ": " << x << ", " << y << "\n";
  }
}

int main(int argc, char **argv) {
  std::vector<std::unique_ptr<rgbd_bridge::RGBDSensor>> devices;
  std::vector<std::string> names;

  std::vector<rgbd_bridge::ImageType> channels = {
      rgbd_bridge::ImageType::RGB,
      rgbd_bridge::ImageType::DEPTH,
      rgbd_bridge::ImageType::RECT_RGB,
      rgbd_bridge::ImageType::RECT_RGB_ALIGNED_DEPTH,
      rgbd_bridge::ImageType::DEPTH_ALIGNED_RGB,
      rgbd_bridge::ImageType::IR};

  for (int i = 0; i < rgbd_bridge::RealSenseSR300::get_number_of_cameras();
       i++) {
    auto real_sense = std::make_unique<rgbd_bridge::RealSenseSR300>(i);
    real_sense->set_mode(rs_ivcam_preset::RS_IVCAM_PRESET_SHORT_RANGE);

    devices.push_back(std::move(real_sense));
    rgbd_bridge::RGBDSensor &test = *devices.back();

    test.Start(channels, rgbd_bridge::ImageType::RECT_RGB);

    for (const auto type : channels) {
      names.push_back(ImageTypeToString(type) + std::to_string(i));
    }
  }
  for (const auto &name : names) {
    cv::namedWindow(name);
  }

  std::vector<std::thread> img_threads;
  std::vector<std::thread> cloud_threads;

  std::vector<std::vector<UserClickedPt>> inputs(
      devices.size(), std::vector<UserClickedPt>(channels.size()));

  for (size_t i = 0; i < devices.size(); i++) {
    for (size_t c = 0; c < channels.size(); c++) {
      inputs[i][c].window_name =
          ImageTypeToString(channels[c]) + std::to_string(i);
    }

    img_threads.emplace_back(
        std::thread(image_loop, devices[i].get(), i, channels, (&inputs[i])));
    for (size_t c = 0; c < channels.size(); c++) {
      cv::setMouseCallback(ImageTypeToString(channels[c]) + std::to_string(i),
                           mouse_click, &(inputs[i][c]));
    }
    cloud_threads.emplace_back(std::thread(cloud_loop, devices[i].get()));
  }

  for (auto &thread : img_threads)
    thread.join();

  for (auto &thread : cloud_threads)
    thread.join();

  return 0;
}