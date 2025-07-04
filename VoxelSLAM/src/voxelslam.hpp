#pragma once

#include "tools.hpp"
#include "ekf_imu.hpp"
#include "voxel_map.hpp"
#include "feature_point.hpp"
#include "loop_refine.hpp"
#include <mutex>
#include <Eigen/Eigenvalues>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <malloc.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <malloc.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <pcl/common/transforms.h> 
#include <execution>
using namespace std;
pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr;
ros::Publisher pub_scan, pub_cmap, pub_init, pub_pmap;
ros::Publisher pub_test, pub_prev_path, pub_curr_path;
ros::Subscriber sub_imu, sub_pcl;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

mutex mBuf;
Features feat;
deque<sensor_msgs::Imu::Ptr> imu_buf;
deque<pcl::PointCloud<PointType>::Ptr> pcl_buf;
deque<double> time_buf;

double imu_last_time = -1;
int point_notime = 0;
double last_pcl_time = -1;

void imu_handler(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  static int flag = 1;
  if(flag)
  {
    flag = 0;
    printf("Time0: %lf\n", msg_in->header.stamp.toSec());
  }
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->linear_acceleration.x = msg->linear_acceleration.x * 9.81;
  msg->linear_acceleration.y = msg->linear_acceleration.y * 9.81;
  msg->linear_acceleration.z = msg->linear_acceleration.z * 9.81;

  // For Hilti 2022 exp03
  // double t0 = 1646320760 + 255.5;
  // double t1 = 1646320760 + 256.2;
  // double tc = msg->header.stamp.toSec();
  // if(tc > t0 && tc < t1)
  //   msg->linear_acceleration.z = -9.7;

  mBuf.lock();
  imu_last_time = msg->header.stamp.toSec();
  imu_buf.push_back(msg);
  mBuf.unlock();
}

template<class T>
void pcl_handler(T &msg)
{
  pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
  double t0 = feat.process(msg, *pl_ptr);

  if(pl_ptr->empty())
  {
    PointType ap; 
    ap.x = 0; ap.y = 0; ap.z = 0; 
    ap.intensity = 0; ap.curvature = 0;
    pl_ptr->push_back(ap);
    ap.curvature = 0.09;
    pl_ptr->push_back(ap);
  }

  sort(pl_ptr->begin(), pl_ptr->end(), [](PointType &x, PointType &y)
  {
    return x.curvature < y.curvature;
  });
  while(pl_ptr->back().curvature > 0.11)
    pl_ptr->points.pop_back();

  mBuf.lock();
  time_buf.push_back(t0);
  pcl_buf.push_back(pl_ptr);
  mBuf.unlock();
}

bool sync_packages(pcl::PointCloud<PointType>::Ptr &pl_ptr, deque<sensor_msgs::Imu::Ptr> &imus, IMUEKF &p_imu)
{
  static bool pl_ready = false;

  if(!pl_ready)
  {
    if(pcl_buf.empty()) return false;

    mBuf.lock();
    pl_ptr = pcl_buf.front();
    p_imu.pcl_beg_time = time_buf.front();
    pcl_buf.pop_front(); time_buf.pop_front();
    mBuf.unlock();

    p_imu.pcl_end_time = p_imu.pcl_beg_time + pl_ptr->back().curvature;

    if(point_notime)
    {
      if(last_pcl_time < 0)
      {
        last_pcl_time = p_imu.pcl_beg_time;
        return false;
      }

      p_imu.pcl_end_time = p_imu.pcl_beg_time;
      p_imu.pcl_beg_time = last_pcl_time;
      last_pcl_time = p_imu.pcl_end_time;
    }

    pl_ready = true;
  }

  if(!pl_ready || imu_last_time <= p_imu.pcl_end_time) return false;

  mBuf.lock();
  double imu_time = imu_buf.front()->header.stamp.toSec();
  while((!imu_buf.empty()) && (imu_time < p_imu.pcl_end_time)) 
  {
    imu_time = imu_buf.front()->header.stamp.toSec();
    if(imu_time > p_imu.pcl_end_time) break;
    imus.push_back(imu_buf.front());
    imu_buf.pop_front();
  }
  mBuf.unlock();

  if(imu_buf.empty())
  {
    printf("imu buf empty\n"); exit(0);
  }

  pl_ready = false;

  if(imus.size() > 4)
    return true;
  else
    return false;
}

double dept_err, beam_err;
void calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var) 
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

// Compute the variance of the each point
void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr, double dept_err, double beam_err)
{
  int plsize = pl_cur.size();
  pptr->clear();
  pptr->resize(plsize);
  for(int i=0; i<plsize; i++)
  {
    PointType &ap = pl_cur[i];
    pointVar &pv = pptr->at(i);
    pv.pnt << ap.x, ap.y, ap.z;
    calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
    pv.pnt = ext.R * pv.pnt + ext.p;
    pv.var = ext.R * pv.var * ext.R.transpose();
  }
}

void pvec_update(PVecPtr pptr, IMUST &x_curr, PLV(3) &pwld)
{
  Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);

  for(pointVar &pv: *pptr)
  {
    Eigen::Matrix3d phat = hat(pv.pnt);
    pv.var = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
    pwld.push_back(x_curr.R * pv.pnt + x_curr.p);
  }
}

double get_memory()
{
  ifstream infile("/proc/self/status");
  double mem = -1;
  string lineStr, str;
  while(getline(infile, lineStr))
  {
    stringstream ss(lineStr);
    bool is_find = false;
    while(ss >> str)
    {
      if(str == "VmRSS:")
      {
        is_find = true; continue;
      }

      if(is_find) mem = stod(str);
      break;
    }
    if(is_find) break;
  }
  return mem / (1048576);
}


