#ifndef LOOP_REFINE_HPP
#define LOOP_REFINE_HPP

#include "tools.hpp"
#include "voxel_map.hpp"
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;

struct ScanPose
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x;
  PVecPtr pvec;
  Eigen::Matrix<double, 6, 1> v6;

  ScanPose(IMUST &_x, PVecPtr _pvec): x(_x), pvec(_pvec)
  {
    v6.setZero();
  }

  void update(IMUST dx)
  {
    x.v = dx.R * x.v;
    x.p = dx.R * x.p + dx.p;
    x.R = dx.R * x.R;
  }


};






#endif
