#include "voxelslam.hpp"

using namespace std;

class ResultOutput
{
public:
  static ResultOutput &instance()
  {
    static ResultOutput inst;
    return inst;
  }

  void pub_odom_func(IMUST &xc)
  {
    Eigen::Quaterniond q_this(xc.R);
    Eigen::Vector3d t_this = xc.p;

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(t_this.x(), t_this.y(), t_this.z()));
    q.setW(q_this.w());
    q.setX(q_this.x());
    q.setY(q_this.y());
    q.setZ(q_this.z());
    transform.setRotation(q);
    ros::Time ct = ros::Time::now();
    br.sendTransform(tf::StampedTransform(transform, ct, "/camera_init", "/aft_mapped"));
  }

  void pub_localtraj(PLV(3) & pwld, double jour, IMUST &x_curr, int cur_session, pcl::PointCloud<PointType> &pcl_path)
  {
    pub_odom_func(x_curr);
    pcl::PointCloud<PointType> pcl_send;
    pcl_send.reserve(pwld.size());
    for (Eigen::Vector3d &pw : pwld)
    {
      Eigen::Vector3d pvec = pw;
      PointType ap;
      ap.x = pvec.x();
      ap.y = pvec.y();
      ap.z = pvec.z();
      pcl_send.push_back(ap);
    }
    pub_pl_func(pcl_send, pub_scan);

    Eigen::Vector3d pcurr = x_curr.p;

    PointType ap;
    ap.x = pcurr[0];
    ap.y = pcurr[1];
    ap.z = pcurr[2];
    ap.curvature = jour;
    ap.intensity = cur_session;
    pcl_path.push_back(ap);
    pub_pl_func(pcl_path, pub_curr_path);
  }

  void pub_localmap(int mgsize, int cur_session, vector<PVecPtr> &pvec_buf, vector<IMUST> &x_buf, pcl::PointCloud<PointType> &pcl_path, int win_base, int win_count)
  {
    pcl::PointCloud<PointType> pcl_send;
    for (int i = 0; i < mgsize; i++)
    {
      for (int j = 0; j < pvec_buf[i]->size(); j += 3)
      {
        pointVar &pv = pvec_buf[i]->at(j);
        Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
        PointType ap;
        ap.x = pvec[0];
        ap.y = pvec[1];
        ap.z = pvec[2];
        ap.intensity = cur_session;
        pcl_send.push_back(ap);
      }
    }

    for (int i = 0; i < win_count; i++)
    {
      Eigen::Vector3d pcurr = x_buf[i].p;
      pcl_path[i + win_base].x = pcurr[0];
      pcl_path[i + win_base].y = pcurr[1];
      pcl_path[i + win_base].z = pcurr[2];
    }

    pub_pl_func(pcl_path, pub_curr_path);
    pub_pl_func(pcl_send, pub_cmap);
  }

  void pub_global_path(vector<vector<ScanPose *> *> &relc_bl_buf, ros::Publisher &pub_relc, vector<int> &ids)
  {
    pcl::PointCloud<pcl::PointXYZI> pl;
    pcl::PointXYZI pp;
    int idsize = ids.size();

    for (int i = 0; i < idsize; i++)
    {
      pp.intensity = ids[i];
      for (ScanPose *bl : *(relc_bl_buf[ids[i]]))
      {
        pp.x = bl->x.p[0];
        pp.y = bl->x.p[1];
        pp.z = bl->x.p[2];
        pl.push_back(pp);
      }
    }
    pub_pl_func(pl, pub_relc);
  }

  void pub_globalmap(vector<vector<Keyframe *> *> &relc_submaps, vector<int> &ids, ros::Publisher &pub)
  {
    pcl::PointCloud<pcl::PointXYZI> pl;
    pub_pl_func(pl, pub);
    pcl::PointXYZI pp;

    uint interval_size = 5e6;
    uint psize = 0;
    for (int id : ids)
    {
      vector<Keyframe *> &smps = *(relc_submaps[id]);
      for (int i = 0; i < smps.size(); i++)
        psize += smps[i]->plptr->size();
    }
    int jump = psize / (10 * interval_size) + 1;

    for (int id : ids)
    {
      pp.intensity = id;
      vector<Keyframe *> &smps = *(relc_submaps[id]);
      for (int i = 0; i < smps.size(); i++)
      {
        IMUST xx = smps[i]->x0;
        for (int j = 0; j < smps[i]->plptr->size(); j += jump)
        // for(int j=0; j<smps[i]->plptr->size(); j+=1)
        {
          PointType &ap = smps[i]->plptr->points[j];
          Eigen::Vector3d vv(ap.x, ap.y, ap.z);
          vv = xx.R * vv + xx.p;
          pp.x = vv[0];
          pp.y = vv[1];
          pp.z = vv[2];
          pl.push_back(pp);
        }

        if (pl.size() > interval_size)
        {
          pub_pl_func(pl, pub);
          sleep(0.05);
          pl.clear();
        }
      }
    }
    pub_pl_func(pl, pub);
  }
};


class Initialization
{
public:
  static Initialization &instance()
  {
    static Initialization inst;
    return inst;
  }

  void align_gravity(vector<IMUST> &xs)
  {
    Eigen::Vector3d g0 = xs[0].g;
    Eigen::Vector3d n0 = g0 / g0.norm();
    Eigen::Vector3d n1(0, 0, 1);
    if (n0[2] < 0)
      n1[2] = -1;

    Eigen::Vector3d rotvec = n0.cross(n1);
    double rnorm = rotvec.norm();
    rotvec = rotvec / rnorm;

    Eigen::AngleAxisd angaxis(asin(rnorm), rotvec);
    Eigen::Matrix3d rot = angaxis.matrix();
    g0 = rot * g0;

    Eigen::Vector3d p0 = xs[0].p;
    for (int i = 0; i < xs.size(); i++)
    {
      xs[i].p = rot * (xs[i].p - p0) + p0;
      xs[i].R = rot * xs[i].R;
      xs[i].v = rot * xs[i].v;
      xs[i].g = g0;
    }
  }

  void motion_blur(pcl::PointCloud<PointType> &pl, PVec &pvec, IMUST xc, IMUST xl, deque<sensor_msgs::Imu::Ptr> &imus, double pcl_beg_time, IMUST &extrin_para)
  {
    xc.bg = xl.bg;
    xc.ba = xl.ba;
    Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(xc.v), pos_imu(xc.p);
    Eigen::Matrix3d R_imu(xc.R);
    vector<IMUST> imu_poses;

    for (auto it_imu = imus.end() - 1; it_imu != imus.begin(); it_imu--)
    {
      sensor_msgs::Imu &head = **(it_imu - 1);
      sensor_msgs::Imu &tail = **(it_imu);

      angvel_avr << 0.5 * (head.angular_velocity.x + tail.angular_velocity.x),
          0.5 * (head.angular_velocity.y + tail.angular_velocity.y),
          0.5 * (head.angular_velocity.z + tail.angular_velocity.z);
      acc_avr << 0.5 * (head.linear_acceleration.x + tail.linear_acceleration.x),
          0.5 * (head.linear_acceleration.y + tail.linear_acceleration.y),
          0.5 * (head.linear_acceleration.z + tail.linear_acceleration.z);

      angvel_avr -= xc.bg;
      acc_avr = acc_avr * imupre_scale_gravity - xc.ba;

      double dt = head.header.stamp.toSec() - tail.header.stamp.toSec();
      Eigen::Matrix3d acc_avr_skew = hat(acc_avr);
      Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

      acc_imu = R_imu * acc_avr + xc.g;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
      R_imu = R_imu * Exp_f;

      double offt = head.header.stamp.toSec() - pcl_beg_time;
      imu_poses.emplace_back(offt, R_imu, pos_imu, vel_imu, angvel_avr, acc_imu);
    }

    pointVar pv;
    pv.var.setIdentity();
    if (point_notime)
    {
      for (PointType &ap : pl.points)
      {
        pv.pnt << ap.x, ap.y, ap.z;
        pv.pnt = extrin_para.R * pv.pnt + extrin_para.p;
        pvec.push_back(pv);
      }
      return;
    }
    auto it_pcl = pl.end() - 1;
    // for(auto it_kp=imu_poses.end(); it_kp!=imu_poses.begin(); it_kp--)
    for (auto it_kp = imu_poses.begin(); it_kp != imu_poses.end(); it_kp++)
    {
      // IMUST &head = *(it_kp - 1);
      IMUST &head = *it_kp;
      R_imu = head.R;
      acc_imu = head.ba;
      vel_imu = head.v;
      pos_imu = head.p;
      angvel_avr = head.bg;

      for (; it_pcl->curvature > head.t; it_pcl--)
      {
        double dt = it_pcl->curvature - head.t;
        Eigen::Matrix3d R_i = R_imu * Exp(angvel_avr, dt);
        Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - xc.p;

        Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        Eigen::Vector3d P_compensate = xc.R.transpose() * (R_i * (extrin_para.R * P_i + extrin_para.p) + T_ei);

        // pv.pnt = P_compensate;
        pv.pnt = P_i;
        pvec.push_back(pv);
        if (it_pcl == pl.begin())
          break;
      }
    }
  }

  int motion_init(vector<pcl::PointCloud<PointType>::Ptr> &pl_origs, vector<deque<sensor_msgs::Imu::Ptr>> &vec_imus, vector<double> &beg_times, Eigen::MatrixXd *hess, LidarFactor &voxhess, vector<IMUST> &x_buf, unordered_map<VOXEL_LOC, OctoTree *> &surf_map, unordered_map<VOXEL_LOC, OctoTree *> &surf_map_slide, vector<PVecPtr> &pvec_buf, int win_size, vector<vector<SlideWindow *>> &sws, IMUST &x_curr, deque<IMU_PRE *> &imu_pre_buf, IMUST &extrin_para)
  {
    PLV(3)
    pwld;
    double last_g_norm = x_buf[0].g.norm();
    int converge_flag = 0;

    double min_eigen_value_orig = min_eigen_value;
    vector<double> eigen_value_array_orig = plane_eigen_value_thre;

    min_eigen_value = 0.02;
    for (double &iter : plane_eigen_value_thre)
      iter = 1.0 / 4;

    double t0 = ros::Time::now().toSec();
    double converge_thre = 0.05;
    int converge_times = 0;
    bool is_degrade = true;
    Eigen::Vector3d eigvalue;
    eigvalue.setZero();

    for (int iterCnt = 0; iterCnt < 10; iterCnt++)
    {
      if (converge_flag == 1)
      {
        min_eigen_value = min_eigen_value_orig;
        plane_eigen_value_thre = eigen_value_array_orig;
      }

      vector<OctoTree *> octos;
      // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      // {
      //   iter->second->tras_ptr(octos);
      //   iter->second->clear_slwd(sws[0]);
      //   delete iter->second;
      // }

      // for (int i = 0; i < octos.size(); i++)
      //   delete octos[i];
      // surf_map.clear();
      // octos.clear();
      // surf_map_slide.clear();

      for (int i = 0; i < win_size; i++)
      {
        pwld.clear();
        pvec_buf[i]->clear();
        int l = i == 0 ? i : i - 1;
        motion_blur(*pl_origs[i], *pvec_buf[i], x_buf[i], x_buf[l], vec_imus[i], beg_times[i], extrin_para);

        if (converge_flag == 1)
        {
          for (pointVar &pv : *pvec_buf[i])
            calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
          pvec_update(pvec_buf[i], x_buf[i], pwld);
        }
        else
        {
          for (pointVar &pv : *pvec_buf[i])
            pwld.push_back(x_buf[i].R * pv.pnt + x_buf[i].p);
        }

        // cut_voxel(surf_map, pvec_buf[i], i, surf_map_slide, win_size, pwld, sws[0]);
      }

      // LidarFactor voxhess(win_size);
      voxhess.clear();
      voxhess.win_size = win_size;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      {

        // iter->second->recut(win_size, x_buf, sws[0]);
        iter->second->tras_opt(voxhess);
      }
      if (voxhess.plvec_voxels.size() < 10)
        break;
      LI_BA_OptimizerGravity opt_lsv;
      vector<double> resis;
      opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, hess, 3);
      Eigen::Matrix3d nnt;
      nnt.setZero();
      printf("%d: %lf %lf %lf: %lf %lf\n", iterCnt, x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm(), fabs(resis[0] - resis[1]) / resis[0]);

      for (int i = 0; i < win_size - 1; i++)
        delete imu_pre_buf[i];
      imu_pre_buf.clear();

      for (int i = 1; i < win_size; i++)
      {
        imu_pre_buf.push_back(new IMU_PRE(x_buf[i - 1].bg, x_buf[i - 1].ba));
        imu_pre_buf.back()->push_imu(vec_imus[i]);
      }
      std::cout << "fabs(resis[0] - resis[1]) / resis[0]:" << fabs(resis[0] - resis[1]) / resis[0] << std::endl;
      if (fabs(resis[0] - resis[1]) / resis[0] < converge_thre && iterCnt >= 2)
      {
        for (Eigen::Matrix3d &iter : voxhess.eig_vectors)
        {
          Eigen::Vector3d v3 = iter.col(0);
          nnt += v3 * v3.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
        eigvalue = saes.eigenvalues();
        is_degrade = eigvalue[0] < 15 ? true : false;
        std::cout << "eigvalue[0]:" << eigvalue[0] << std::endl;

        converge_thre = 0.01;
        if (converge_flag == 0)
        {
          std::cout << "converge_flag：" << converge_flag << std::endl;
          align_gravity(x_buf);
          converge_flag = 1;
          continue;
        }
        else
          break;
      }
    }
    std::cout << "converge_flag:" << converge_flag << std::endl;
    std::cout << "x_curr.g:" << x_curr.g << std::endl;
    x_curr = x_buf[win_size - 1];
    double gnm = x_curr.g.norm();
    std::cout << "gnm:" << gnm << std::endl;
    std::cout << "is_degrade:" << is_degrade << std::endl;
    if (is_degrade || gnm < 9.5 || gnm > 10.0)
    {
      converge_flag = 0;
    }
    if (converge_flag == 0)
    {
      // vector<OctoTree *> octos;
      // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      // {
      //   iter->second->tras_ptr(octos);
      //   iter->second->clear_slwd(sws[0]);
      //   delete iter->second;
      // }
      // for (int i = 0; i < octos.size(); i++)
      // delete octos[i];
      // surf_map.clear();
      // octos.clear();
      // surf_map_slide.clear();
    }

    printf("mn: %lf %lf %lf\n", eigvalue[0], eigvalue[1], eigvalue[2]);
    Eigen::Vector3d angv(vec_imus[0][0]->angular_velocity.x, vec_imus[0][0]->angular_velocity.y, vec_imus[0][0]->angular_velocity.z);
    Eigen::Vector3d acc(vec_imus[0][0]->linear_acceleration.x, vec_imus[0][0]->linear_acceleration.y, vec_imus[0][0]->linear_acceleration.z);
    acc *= 9.8;

    pl_origs.clear();
    vec_imus.clear();
    beg_times.clear();
    double t1 = ros::Time::now().toSec();
    printf("init time: %lf\n", t1 - t0);

    // align_gravity(x_buf);
    pcl::PointCloud<PointType> pcl_send;
    PointType pt;
    for (int i = 0; i < win_size; i++)
      for (pointVar &pv : *pvec_buf[i])
      {
        Eigen::Vector3d vv = x_buf[i].R * pv.pnt + x_buf[i].p;
        pt.x = vv[0];
        pt.y = vv[1];
        pt.z = vv[2];
        pcl_send.push_back(pt);
      }
    pub_pl_func(pcl_send, pub_init);
    std::cout << "final converge_flag:" << converge_flag << std::endl;
    return converge_flag;
  }
};

class VOXEL_SLAM
{
public:
  pcl::PointCloud<PointType> pcl_path;
  IMUST x_curr, extrin_para;
  IMUEKF odom_ekf;
  unordered_map<VOXEL_LOC, OctoTree *> surf_map, surf_map_slide, surf_map_static;
  double down_size;

  int win_size;
  vector<IMUST> x_buf;
  vector<PVecPtr> pvec_buf;
  deque<IMU_PRE *> imu_pre_buf;
  int win_count = 0, win_base = 0;
  vector<vector<SlideWindow *>> sws, sws_static;

  vector<ScanPose *> *scanPoses;


  vector<Keyframe *> *keyframes;

  IMUST dx;
  pcl::PointCloud<PointType>::Ptr pl_kdmap;
  pcl::KdTreeFLANN<PointType> kd_keyframes;
  int history_kfsize = 0;
  vector<OctoTree *> octos_release;
  int reset_flag = 0;
  int g_update = 0;
  int thread_num = 5;
  int degrade_bound = 10;


  bool is_finish = false;

  vector<string> sessionNames;
  string bagname, savepath;
  int is_save_map;

  pcl::PointCloud<PointType>::Ptr map_pc;

  VOXEL_SLAM(ros::NodeHandle &n)
  {
    map_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>());
    double cov_gyr, cov_acc, rand_walk_gyr, rand_walk_acc;
    vector<double> vecR(9), vecT(3);
    scanPoses = new vector<ScanPose *>();
    keyframes = new vector<Keyframe *>();

    string lid_topic, imu_topic;
    n.param<string>("General/lid_topic", lid_topic, "/livox/lidar");
    n.param<string>("General/imu_topic", imu_topic, "/livox/imu");
    n.param<string>("General/bagname", bagname, "site3_handheld_4");
    n.param<string>("General/save_path", savepath, "");
    n.param<int>("General/lidar_type", feat.lidar_type, 0);
    n.param<double>("General/blind", feat.blind, 0.1);
    n.param<int>("General/point_filter_num", feat.point_filter_num, 3);
    n.param<vector<double>>("General/extrinsic_tran", vecT, vector<double>());
    n.param<vector<double>>("General/extrinsic_rota", vecR, vector<double>());
    n.param<int>("General/is_save_map", is_save_map, 0);

    sub_imu = n.subscribe(imu_topic, 80000, imu_handler);
    if (feat.lidar_type == LIVOX)
      sub_pcl = n.subscribe<livox_ros_driver::CustomMsg>(lid_topic, 1, pcl_handler);
    else
      sub_pcl = n.subscribe<sensor_msgs::PointCloud2>(lid_topic, 1, pcl_handler);
    odom_ekf.imu_topic = imu_topic;

    n.param<double>("Odometry/cov_gyr", cov_gyr, 0.1);
    n.param<double>("Odometry/cov_acc", cov_acc, 0.1);
    n.param<double>("Odometry/rdw_gyr", rand_walk_gyr, 1e-4);
    n.param<double>("Odometry/rdw_acc", rand_walk_acc, 1e-4);
    n.param<double>("Odometry/down_size", down_size, 0.1);
    n.param<double>("Odometry/dept_err", dept_err, 0.02);
    n.param<double>("Odometry/beam_err", beam_err, 0.05);
    n.param<double>("Odometry/voxel_size", voxel_size, 1);
    n.param<double>("Odometry/min_eigen_value", min_eigen_value, 0.0025);
    n.param<int>("Odometry/degrade_bound", degrade_bound, 10);
    n.param<int>("Odometry/point_notime", point_notime, 0);
    odom_ekf.point_notime = point_notime;

    feat.blind = feat.blind * feat.blind;
    odom_ekf.cov_gyr << cov_gyr, cov_gyr, cov_gyr;
    odom_ekf.cov_acc << cov_acc, cov_acc, cov_acc;
    odom_ekf.cov_bias_gyr << rand_walk_gyr, rand_walk_gyr, rand_walk_gyr;
    odom_ekf.cov_bias_acc << rand_walk_acc, rand_walk_acc, rand_walk_acc;
    odom_ekf.Lid_offset_to_IMU << vecT[0], vecT[1], vecT[2];
    odom_ekf.Lid_rot_to_IMU << vecR[0], vecR[1], vecR[2],
        vecR[3], vecR[4], vecR[5],
        vecR[6], vecR[7], vecR[8];
    extrin_para.R = odom_ekf.Lid_rot_to_IMU;
    extrin_para.p = odom_ekf.Lid_offset_to_IMU;
    min_point << 5, 5, 5, 5;

    n.param<int>("LocalBA/win_size", win_size, 10);
    n.param<int>("LocalBA/max_layer", max_layer, 2);
    n.param<double>("LocalBA/cov_gyr", cov_gyr, 0.1);
    n.param<double>("LocalBA/cov_acc", cov_acc, 0.1);
    n.param<double>("LocalBA/rdw_gyr", rand_walk_gyr, 1e-4);
    n.param<double>("LocalBA/rdw_acc", rand_walk_acc, 1e-4);
    n.param<int>("LocalBA/min_ba_point", min_ba_point, 20);
    n.param<vector<double>>("LocalBA/plane_eigen_value_thre", plane_eigen_value_thre, vector<double>({1, 1, 1, 1}));
    n.param<double>("LocalBA/imu_coef", imu_coef, 1e-4);
    n.param<int>("LocalBA/thread_num", thread_num, 5);

    for (double &iter : plane_eigen_value_thre)
      iter = 1.0 / iter;
    // for(double &iter: plane_eigen_value_thre) iter = 1.0 / iter;

    noiseMeas.setZero();
    noiseWalk.setZero();
    noiseMeas.diagonal() << cov_gyr, cov_gyr, cov_gyr,
        cov_acc, cov_acc, cov_acc;
    noiseWalk.diagonal() << rand_walk_gyr, rand_walk_gyr, rand_walk_gyr,
        rand_walk_acc, rand_walk_acc, rand_walk_acc;

    int ss = 0;
    if (access((savepath + bagname + "/").c_str(), X_OK) == -1)
    {
      string cmd = "mkdir " + savepath + bagname + "/";
      ss = system(cmd.c_str());
    }
    else
      ss = -1;

    if (ss != 0 && is_save_map == 1)
    {
      printf("The pointcloud will be saved in this run.\n");
      printf("So please clear or rename the existed folder.\n");
      exit(0);
    }

    sws.resize(thread_num);
    cout << "bagname: " << bagname << endl;

    // 1. 读入 PCD
    std::string map_path = "/home/zc/map/2025.pcd";
    map_pc.reset(new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile(map_path, *map_pc);
    // 2. 下采样（可选，减少点数）
    down_sampling_voxel(*map_pc, 0.05); // 你已有的 down_sampling_voxel
    std::cout<<"map_pc->point.size():"<<map_pc->points.size()<<std::endl;

    // 3. 分配给 prior_pvec 和 pwld
    PVecPtr prior_pvec(new PVec());
    PLV(3)
    pwld;
    prior_pvec->reserve(map_pc->size());
    pwld.reserve(map_pc->size());
    IMUST T;
    T.R.setIdentity();
    T.p.setZero();
    var_init(T, *map_pc, prior_pvec, 0.005, 0.005);

    for (auto &pt : map_pc->points)
    {
      // 把 PCD 点当成 pointVar
      pointVar pv;
      pv.pnt = Eigen::Vector3d(pt.x, pt.y, pt.z);

      // 简单起见，把协方差设为常量 * 单位阵
      // 真实场景里也可做法线估计+KNN-PCA来求每点的局部协方差
      // pv.var = Eigen::Matrix3d::Identity() * 1e-3;

      // prior_pvec->push_back(pv);

      // 世界坐标就直接等于它本身
      pwld.push_back(pv.pnt);
    }

    // 4. 第一次调用 cut_voxel_multi，用来构建八叉树
    sws_static.resize(thread_num);
    for (int i = 0; i < thread_num; ++i)
      sws_static[i].clear();
    cut_voxel_multi(
        /*feat_map=*/surf_map_static,
        /*pvec=*/prior_pvec,
        /*win_count=*区分静态地图点和滑窗点*/ -1,
        /*feat_tem_map=*/surf_map_slide,
        /*wdsize=*/win_size,
        /*pwld=*/pwld,
        /*sws=*/sws_static);

    vector<IMUST> dummy_xbuf(1);
    dummy_xbuf[0].R = Eigen::Matrix3d::Identity();
    dummy_xbuf[0].p = Eigen::Vector3d::Zero();
    vector<vector<SlideWindow *>> dummy_sws(1);
    dummy_sws[0] = sws_static[0]; // 内容随便，只要非空即可

    // 5) 对 surf_map 做一次 recut，计算每个节点的 eig
    LidarFactor voxhess(win_size);
    multi_recut(surf_map_static, // 用你的 static_map
                /*win_count=*/1,
                dummy_xbuf,
                /*voxopt=*/voxhess, // 这里的 voxopt 只是暂存 eig，不用清
                sws_static);

    // 6) 遍历所有节点，一次性调用 plane_update()
    function<void(OctoTree *)> init_plane = [&](OctoTree *node)
    {
      if (node->octo_state == 0 && node->layer >= 0 && node->plane.is_plane)
      {
        node->plane_update();
      }
      for (int i = 0; i < 8; i++)
      {
        if (node->leaves[i])
          init_plane(node->leaves[i]);
      }
    };
    for (auto &kv : surf_map_static)
    {
      init_plane(kv.second);
    }

    std::cout << "load map finish" << std::endl;
  }

  // 深拷贝整个 surf_map
  // void deepCopySurfMap(
  //     const std::unordered_map<VOXEL_LOC, OctoTree *> &original, std::unordered_map<VOXEL_LOC, OctoTree *> &copy)
  // {
  //   copy.reserve(original.size());
  //   for (auto &kv : original)
  //   {
  //     const VOXEL_LOC &key = kv.first;
  //     const OctoTree *node = kv.second;
  //     copy.emplace(key, node->cloneOctoTree());
  //   }
  //   return;
  // }
  void deepCopySurfMap(
      const std::unordered_map<VOXEL_LOC, OctoTree *> &original,
      std::unordered_map<VOXEL_LOC, OctoTree *> &copy)
  {
    // 1) 将 map 拷贝到 vector 中
    std::vector<std::pair<VOXEL_LOC, OctoTree *>> items;
    items.reserve(original.size());
    for (auto &kv : original)
    {
      items.emplace_back(kv.first, kv.second);
    }

    // 2) 并行 clone
    std::vector<std::pair<VOXEL_LOC, OctoTree *>> clones(items.size());
    std::transform(
        std::execution::par,
        items.begin(), items.end(),
        clones.begin(),
        [](auto &kv)
        {
          // kv.first：VOXEL_LOC，kv.second：原节点指针
          // cloneOctoTree 必须是线程安全的（不访问全局状态）
          return std::make_pair(kv.first, kv.second->cloneOctoTree());
        });

    // 3) 批量插入到 unordered_map（单线程安全）
    copy.reserve(clones.size());
    for (auto &kv : clones)
    {
      copy.emplace(std::move(kv.first), kv.second);
    }
  }

  // The point-to-plane alignment for odometry
  bool lio_state_estimation(PVecPtr pptr)
  {
    IMUST x_prop = x_curr;

    const int num_max_iter = 4;
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    int rematch_num = 0;
    int match_num = 0;

    int psize = pptr->size();
    vector<OctoTree *> octos;
    octos.resize(psize, nullptr);

    Eigen::Matrix3d nnt;
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();
    for (int iterCount = 0; iterCount < num_max_iter; iterCount++)
    {
      Eigen::Matrix<double, 6, 6> HTH;
      HTH.setZero();
      Eigen::Matrix<double, 6, 1> HTz;
      HTz.setZero();
      Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
      Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);
      match_num = 0;
      nnt.setZero();

      for (int i = 0; i < psize; i++)
      {
        pointVar &pv = pptr->at(i);
        Eigen::Matrix3d phat = hat(pv.pnt);
        Eigen::Matrix3d var_world = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        double sigma_d = 0;
        Plane *pla = nullptr;
        int flag = 0;
        if (octos[i] != nullptr && octos[i]->inside(wld))
        {
          double max_prob = 0;
          flag = octos[i]->match(wld, pla, max_prob, var_world, sigma_d, octos[i]);
        }
        else
        {
          flag = match(surf_map, wld, pla, var_world, sigma_d, octos[i]);
        }

        if (flag)
        // if(pla != nullptr)
        {
          Plane &pp = *pla;
          double R_inv = 1.0 / (0.0005 + sigma_d);
          double resi = pp.normal.dot(wld - pp.center);

          Eigen::Matrix<double, 6, 1> jac;
          jac.head(3) = phat * x_curr.R.transpose() * pp.normal;
          jac.tail(3) = pp.normal;
          HTH += R_inv * jac * jac.transpose();
          HTz -= R_inv * jac * resi;
          nnt += pp.normal * pp.normal.transpose();
          match_num++;
        }
      }

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);

      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      EKF_stop_flg = false;
      flg_EKF_converged = false;

      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015))
        flg_EKF_converged = true;

      if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == num_max_iter - 2)))
      {
        rematch_num++;
      }

      if (rematch_num >= 2 || (iterCount == num_max_iter - 1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if (EKF_stop_flg)
        break;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
    Eigen::Vector3d evalue = saes.eigenvalues();
    // printf("eva %d: %lf\n", match_num, evalue[0]);

    if (evalue[0] < 14)
      return false;
    else
      return true;
  }

  // The point-to-plane alignment for initialization
  pcl::PointCloud<PointType>::Ptr pl_tree;
  void lio_state_estimation_kdtree(PVecPtr pptr)
  {
    static pcl::KdTreeFLANN<PointType> kd_map;
    if (pl_tree->size() < 100)
    {
      for (pointVar pv : *pptr)
      {
        PointType pp;
        pv.pnt = x_curr.R * pv.pnt + x_curr.p;
        pp.x = pv.pnt[0];
        pp.y = pv.pnt[1];
        pp.z = pv.pnt[2];
        pl_tree->push_back(pp);
      }
      kd_map.setInputCloud(pl_tree);
      return;
    }

    const int num_max_iter = 4;
    IMUST x_prop = x_curr;
    int psize = pptr->size();
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    double max_dis = 2 * 2;
    vector<float> sqdis(NMATCH);
    vector<int> nearInd(NMATCH);
    PLV(3)
    vecs(NMATCH);
    int rematch_num = 0;
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();

    Eigen::Matrix<double, NMATCH, 1> b;
    b.setOnes();
    b *= -1.0f;

    vector<double> ds(psize, -1);
    PLV(3)
    directs(psize);
    bool refind = true;

    for (int iterCount = 0; iterCount < num_max_iter; iterCount++)
    {
      Eigen::Matrix<double, 6, 6> HTH;
      HTH.setZero();
      Eigen::Matrix<double, 6, 1> HTz;
      HTz.setZero();
      int valid = 0;
      for (int i = 0; i < psize; i++)
      {
        pointVar &pv = pptr->at(i);
        Eigen::Matrix3d phat = hat(pv.pnt);
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        if (refind)
        {
          PointType apx;
          apx.x = wld[0];
          apx.y = wld[1];
          apx.z = wld[2];
          kd_map.nearestKSearch(apx, NMATCH, nearInd, sqdis);

          Eigen::Matrix<double, NMATCH, 3> A;
          for (int i = 0; i < NMATCH; i++)
          {
            PointType &pp = pl_tree->points[nearInd[i]];
            A.row(i) << pp.x, pp.y, pp.z;
          }
          Eigen::Vector3d direct = A.colPivHouseholderQr().solve(b);
          bool check_flag = false;
          for (int i = 0; i < NMATCH; i++)
          {
            if (fabs(direct.dot(A.row(i)) + 1.0) > 0.1)
              check_flag = true;
          }

          if (check_flag)
          {
            ds[i] = -1;
            continue;
          }

          double d = 1.0 / direct.norm();
          // direct *= d;
          ds[i] = d;
          directs[i] = direct * d;
        }

        if (ds[i] >= 0)
        {
          double pd2 = directs[i].dot(wld) + ds[i];
          Eigen::Matrix<double, 6, 1> jac_s;
          jac_s.head(3) = phat * x_curr.R.transpose() * directs[i];
          jac_s.tail(3) = directs[i];

          HTH += jac_s * jac_s.transpose();
          HTz += jac_s * (-pd2);
          valid++;
        }
      }

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv / 1000).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);

      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      refind = false;
      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015))
      {
        refind = true;
        flg_EKF_converged = true;
        rematch_num++;
      }

      if (iterCount == num_max_iter - 2 && !flg_EKF_converged)
      {
        refind = true;
      }

      if (rematch_num >= 2 || (iterCount == num_max_iter - 1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if (EKF_stop_flg)
        break;
    }

    double tt1 = ros::Time::now().toSec();
    for (pointVar pv : *pptr)
    {
      pv.pnt = x_curr.R * pv.pnt + x_curr.p;
      PointType ap;
      ap.x = pv.pnt[0];
      ap.y = pv.pnt[1];
      ap.z = pv.pnt[2];
      pl_tree->push_back(ap);
    }
    down_sampling_voxel(*pl_tree, 0.5);
    kd_map.setInputCloud(pl_tree);
    double tt2 = ros::Time::now().toSec();
  }


  // load the previous keyframe in the local voxel map
  void keyframe_loading(double jour)
  {
    if (history_kfsize <= 0)
      return;
    double tt1 = ros::Time::now().toSec();
    PointType ap_curr;
    ap_curr.x = x_curr.p[0];
    ap_curr.y = x_curr.p[1];
    ap_curr.z = x_curr.p[2];
    vector<int> vec_idx;
    vector<float> vec_dis;
    kd_keyframes.radiusSearch(ap_curr, 10, vec_idx, vec_dis);

    for (int id : vec_idx)
    {
      int ord_kf = pl_kdmap->points[id].curvature;
      if (keyframes->at(id)->exist)
      {
        Keyframe &kf = *(keyframes->at(id));
        IMUST &xx = kf.x0;
        PVec pvec;
        pvec.reserve(kf.plptr->size());

        pointVar pv;
        pv.var.setZero();
        int plsize = kf.plptr->size();
        // for(int j=0; j<plsize; j+=2)
        for (int j = 0; j < plsize; j++)
        {
          PointType ap = kf.plptr->points[j];
          pv.pnt << ap.x, ap.y, ap.z;
          pv.pnt = xx.R * pv.pnt + xx.p;
          pvec.push_back(pv);
        }

        // cut_voxel(surf_map, pvec, win_size, jour);
        kf.exist = 0;
        history_kfsize--;
        break;
      }
    }
  }

  int initialization(deque<sensor_msgs::Imu::Ptr> &imus, Eigen::MatrixXd &hess, LidarFactor &voxhess, PLV(3) & pwld, pcl::PointCloud<PointType>::Ptr pcl_curr)
  {
    static vector<pcl::PointCloud<PointType>::Ptr> pl_origs;
    static vector<double> beg_times;
    static vector<deque<sensor_msgs::Imu::Ptr>> vec_imus;

    pcl::PointCloud<PointType>::Ptr orig(new pcl::PointCloud<PointType>(*pcl_curr));
    if (odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
      return 0;

    if (win_count == 0)
      imupre_scale_gravity = odom_ekf.scale_gravity;

    PVecPtr pptr(new PVec);
    double downkd = down_size >= 0.5 ? down_size : 0.5;
    down_sampling_voxel(*pcl_curr, downkd);
    var_init(extrin_para, *pcl_curr, pptr, dept_err, beam_err);
    lio_state_estimation_kdtree(pptr);

    pwld.clear();
    pvec_update(pptr, x_curr, pwld);

    win_count++;
    x_buf.push_back(x_curr);
    pvec_buf.push_back(pptr);
    ResultOutput::instance().pub_localtraj(pwld, 0, x_curr, sessionNames.size() - 1, pcl_path);

    if (win_count > 1)
    {
      imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count - 2].bg, x_buf[win_count - 2].ba));
      imu_pre_buf[win_count - 2]->push_imu(imus);
    }

    pcl::PointCloud<PointType> pl_mid = *orig;
    down_sampling_close(*orig, down_size);
    if (orig->size() < 1000)
    {
      *orig = pl_mid;
      down_sampling_close(*orig, down_size / 2);
    }

    sort(orig->begin(), orig->end(), [](PointType &x, PointType &y)
         { return x.curvature < y.curvature; });

    pl_origs.push_back(orig);
    beg_times.push_back(odom_ekf.pcl_beg_time);
    vec_imus.push_back(imus);

    int is_success = 0;
    if (win_count >= win_size)
    {
      is_success = Initialization::instance().motion_init(pl_origs, vec_imus, beg_times, &hess, voxhess, x_buf, surf_map_static, surf_map_slide, pvec_buf, win_size, sws, x_curr, imu_pre_buf, extrin_para);
      std::cout << "is_success:" << is_success << std::endl;
      if (is_success == 0)
        return -1;
      return 1;
    }
    return 0;
  }

  void system_reset(deque<sensor_msgs::Imu::Ptr> &imus)
  {
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
      iter->second->tras_ptr(octos_release);
      iter->second->clear_slwd(sws[0]);
      delete iter->second;
    }
    surf_map.clear();
    surf_map_slide.clear();

    x_curr.setZero();
    x_curr.p = Eigen::Vector3d(0, 0, 30);
    odom_ekf.mean_acc.setZero();
    odom_ekf.init_num = 0;
    odom_ekf.IMU_init(imus);
    x_curr.g = -odom_ekf.mean_acc * imupre_scale_gravity;

    for (int i = 0; i < imu_pre_buf.size(); i++)
      delete imu_pre_buf[i];
    x_buf.clear();
    pvec_buf.clear();
    imu_pre_buf.clear();
    pl_tree->clear();

    for (int i = 0; i < win_size; i++)
      mp[i] = i;
    win_base = 0;
    win_count = 0;
    pcl_path.clear();
    pub_pl_func(pcl_path, pub_cmap);
    ROS_WARN("Reset");
  }

  // After local BA, update the map and marginalize the points of oldest scan
  // multi means multiple thread
  void multi_margi(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, double jour, int win_count, vector<IMUST> &xs, LidarFactor &voxopt, vector<SlideWindow *> &sw)
  {
    // for(auto iter=feat_map.begin(); iter!=feat_map.end();)
    // {
    //   iter->second->jour = jour;
    //   iter->second->margi(win_count, 1, xs, voxopt);
    //   if(iter->second->isexist)
    //     iter++;
    //   else
    //   {
    //     iter->second->clear_slwd(sw);
    //     feat_map.erase(iter++);
    //   }
    // }
    // return;

    int thd_num = thread_num;
    vector<vector<OctoTree *> *> octs;
    for (int i = 0; i < thd_num; i++)
      octs.push_back(new vector<OctoTree *>());

    int g_size = feat_map.size();
    if (g_size < thd_num)
      return;
    vector<thread *> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
    {
      iter->second->jour = jour;
      octs[cnt]->push_back(iter->second);
      if (octs[cnt]->size() >= part && cnt < thd_num - 1)
        cnt++;
    }

    auto margi_func = [](int win_cnt, vector<OctoTree *> *oct, vector<IMUST> xxs, LidarFactor &voxhess)
    {
      for (OctoTree *oc : *oct)
      {
        oc->margi(win_cnt, 1, xxs, voxhess);
      }
    };

    for (int i = 1; i < thd_num; i++)
    {
      mthreads[i] = new thread(margi_func, win_count, octs[i], xs, ref(voxopt));
    }

    for (int i = 0; i < thd_num; i++)
    {
      if (i == 0)
      {
        margi_func(win_count, octs[i], xs, voxopt);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    for (auto iter = feat_map.begin(); iter != feat_map.end();)
    {
      // 最旧的帧对应的体素标记为 isexist = false
      if (iter->second->isexist)
        iter++;
      else
      {
        iter->second->clear_slwd(sw);
        feat_map.erase(iter++);
      }
    }

    for (int i = 0; i < thd_num; i++)
      delete octs[i];
  }

  // Determine the plane and recut the voxel map in octo-tree
  void multi_recut(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, int win_count, vector<IMUST> &xs, LidarFactor &voxopt, vector<vector<SlideWindow *>> &sws)
  {
    // for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
    // {
    //   iter->second->recut(win_count, xs, sws[0]);
    //   iter->second->tras_opt(voxopt);
    // }

    int thd_num = thread_num;
    vector<vector<OctoTree *>> octss(thd_num);
    int g_size = feat_map.size();
    if (g_size < thd_num)
      return;
    vector<thread *> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
    {
      octss[cnt].push_back(iter->second);
      if (octss[cnt].size() >= part && cnt < thd_num - 1)
        cnt++;
    }

    auto recut_func = [](int win_count, vector<OctoTree *> &oct, vector<IMUST> xxs, vector<SlideWindow *> &sw)
    {
      for (OctoTree *oc : oct)
        oc->recut(win_count, xxs, sw);
    };

    for (int i = 1; i < thd_num; i++)
    {
      mthreads[i] = new thread(recut_func, win_count, ref(octss[i]), xs, ref(sws[i]));
    }

    for (int i = 0; i < thd_num; i++)
    {
      if (i == 0)
      {
        recut_func(win_count, octss[i], xs, sws[i]);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    for (int i = 1; i < sws.size(); i++)
    {
      sws[0].insert(sws[0].end(), sws[i].begin(), sws[i].end());
      sws[i].clear();
    }

    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
      iter->second->tras_opt(voxopt);
  }

  // The main thread of odometry and local mapping
  void thd_odometry_localmapping(ros::NodeHandle &n)
  {
    PLV(3)
    pwld;
    std::deque<PLV(3)> pwlds;
    std::deque<PVecPtr> pvecptrs;
    double down_sizes[3] = {0.1, 0.2, 0.4};
    Eigen::Vector3d last_pos(0, 0, 0);
    double jour = 0;
    int counter = 0;

    pcl::PointCloud<PointType>::Ptr pcl_curr(new pcl::PointCloud<PointType>());
    int motion_init_flag = 1;
    pl_tree.reset(new pcl::PointCloud<PointType>());
    vector<pcl::PointCloud<PointType>::Ptr> pl_origs;
    vector<double> beg_times;
    vector<deque<sensor_msgs::Imu::Ptr>> vec_imus;
    bool release_flag = false;
    int degrade_cnt = 0;
    LidarFactor voxhess(win_size);
    const int mgsize = 1;
    Eigen::MatrixXd hess;
    while (n.ok())
    {
      ros::spinOnce();


      n.param<bool>("finish", is_finish, false);
      if (is_finish)
      {
        break;
      }

      deque<sensor_msgs::Imu::Ptr> imus;
      if (!sync_packages(pcl_curr, imus, odom_ekf))
      {
        if (octos_release.size() != 0)
        {
          int msize = octos_release.size();
          if (msize > 1000)
            msize = 1000;
          for (int i = 0; i < msize; i++)
          {
            delete octos_release.back();
            octos_release.pop_back();
          }
          malloc_trim(0);
        }
        else if (release_flag)
        {
          release_flag = false;
          vector<OctoTree *> octos;
          for (auto iter = surf_map.begin(); iter != surf_map.end();)
          {
            int dis = jour - iter->second->jour;
            if (dis < 700)
            // if(dis < 200)
            {
              iter++;
            }
            else
            {
              octos.push_back(iter->second);
              iter->second->tras_ptr(octos);
              surf_map.erase(iter++);
            }
          }
          int ocsize = octos.size();
          for (int i = 0; i < ocsize; i++)
            delete octos[i];
          octos.clear();
          malloc_trim(0);
        }
        else if (sws[0].size() > 10000)
        {
          for (int i = 0; i < 500; i++)
          {
            delete sws[0].back();
            sws[0].pop_back();
          }
          malloc_trim(0);
        }

        sleep(0.001);
        continue;
      }
      static int first_flag = 1;
      if (first_flag)
      {
        pcl::PointCloud<PointType> pl;
        pub_pl_func(pl, pub_pmap);
        pub_pl_func(pl, pub_prev_path);
        first_flag = 0;
      }

      double t0 = std::chrono::duration<double, std::milli>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
      double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0;

      if (motion_init_flag)
      {
        // 0603
        x_curr.p = Eigen::Vector3d(11.5402, 12.5136, 0.05);
        Eigen::Matrix3d R;
        R << 0.032015, -0.999461, 0.00720392,
            0.999397, 0.0319141, -0.0136993,
            0.013462, 0.00763815, 0.99988;
        // laserscan3d
        //  x_curr.p = Eigen::Vector3d(6.88, 10.85, -0.2);
        //  Eigen::Matrix3d R;
        //  R << 0.991824, -0.126672, -0.0154664,
        //  0.126695, 0.991942  , 0.000484272,
        //  0.0152804, -0.00243982 ,  0.99988 ;
        // 0430
        //  x_curr.p = Eigen::Vector3d(18.6758, -6.89524, 0);
        //  Eigen::Matrix3d R;
        //  R << -0.0760827 , 0.997075 , 0.00722731 ,
        //  -0.997055 , -0.0761469  , 0.00906114 ,
        //  0.00958497, -0.00651663 ,   0.999933 ;
        // 0630
        //  x_curr.p = Eigen::Vector3d(16.1143, 14.3976, 0.15);
        //  Eigen::Matrix3d R;
        //  R << -0.16833, -0.985633, 0.0139113,
        //  0.985698, -0.168423  , -0.00581409,
        //  0.00807353, 0.0127336 ,  0.999886 ;

        x_curr.R = R;
        std::cout << "x_curr.p" << x_curr.p << std::endl;
        std::cout << "x_curr.R" << x_curr.R << std::endl;
        int init = initialization(imus, hess, voxhess, pwld, pcl_curr);
        std::cout << "init:" << init << std::endl;

        if (init == 1)
        {
          motion_init_flag = 0;
          std::cout << "motion_init_flag:" << motion_init_flag << std::endl;
        }
        else
        {
          if (init == -1)
            system_reset(imus);
          continue;
        }
      }
      else
      {
        if (odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
          continue;

        pcl::PointCloud<PointType> pl_down = *pcl_curr;
        down_sampling_voxel(pl_down, down_size);

        if (pl_down.size() < 500)
        {
          pl_down = *pcl_curr;
          down_sampling_voxel(pl_down, down_size / 2);
        }

        PVecPtr pptr(new PVec);
        var_init(extrin_para, pl_down, pptr, dept_err, beam_err);

        if (lio_state_estimation(pptr))
        {
          if (degrade_cnt > 0)
            degrade_cnt--;
        }
        else
          degrade_cnt++;

        pwld.clear();
        pvec_update(pptr, x_curr, pwld);
        // pwlds.push_back(pwld);
        // pvecptrs.push_back(pptr);
        if (pwlds.size() > win_size)
        {
          pwlds.pop_front();
          pvecptrs.pop_front();
        }

        ResultOutput::instance().pub_localtraj(pwld, jour, x_curr, sessionNames.size() - 1, pcl_path);

        t1 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();

        win_count++;
        x_buf.push_back(x_curr);
        pvec_buf.push_back(pptr);
        if (win_count > 1)
        {
          imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count - 2].bg, x_buf[win_count - 2].ba));
          imu_pre_buf[win_count - 2]->push_imu(imus);
        }

        keyframe_loading(jour);
        voxhess.clear();
        voxhess.win_size = win_size;

        // unordered_map<VOXEL_LOC, OctoTree *> temp_surf_map;
        // temp_surf_map = surf_map;
        // // std::cout<<"temp_surf_map.size():"<<temp_surf_map.size()<<std::endl;
        // // std::cout<<"pvec_buf.size():"<<pvec_buf.size()<<std::endl;
        // // std::cout<<"x_buf.size():"<<x_buf.size()<<std::endl;
        // PVecPtr temp_pvec(new PVec);
        // PLV(3) temp_pwld;
        // for(int a = 0; a < pwlds.size(); a++)
        // {

        //   for(int b = 0; b < pwlds[a].size(); b++)
        //   {
        //     temp_pwld.push_back(pwlds[a][b]);

        //   }
        //   PVecPtr pvec_ptr = pvecptrs[a];
        //   int n = 0;
        //   for (const pointVar &pv : *pvec_ptr) {
        //     pointVar point;
        //     point.pnt = x_curr.R.inverse() * (pwlds[a][n] - x_curr.p);
        //     point.var = pv.var;
        //     n++;
        //     temp_pvec->push_back(point);
        // }
        // }

        vector<OctoTree *> octos;
        vector<SlideWindow *> sws_temp;

        for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        {
          iter->second->tras_ptr(octos);
          iter->second->clear_slwd(sws_temp);
          delete iter->second;
        }

        for (int i = 0; i < octos.size(); i++)
          delete octos[i];
        octos.clear();

        for (int i = 0; i < sws_temp.size(); i++)
          delete sws_temp[i];
        sws_temp.clear();
        malloc_trim(0);
        surf_map.clear();
        surf_map_slide.clear();

        deepCopySurfMap(surf_map_static, surf_map);

        // 把滑窗内所有帧都插入到 surf_map 中
        //    这样 BA 时既有全局先验，又保留了滑窗内的多帧约束
        //   auto before = std::chrono::duration<double, std::milli>(
        //     std::chrono::system_clock::now().time_since_epoch()
        // ).count();
        for (int i = 0; i < win_count; ++i)
        {
          // 先把第 i 帧的点从局部坐标变到世界坐标
          PLV(3)
          temp_pwld;
          pwld.reserve(pvec_buf[i]->size());
          for (const pointVar &pv : *pvec_buf[i])
          {
            temp_pwld.push_back(x_buf[i].R * pv.pnt + x_buf[i].p);
          }
          // idx 参数选择 “滑窗内的全局帧号”
          int frame_idx = i;
          cut_voxel(surf_map, pvec_buf[i], frame_idx, surf_map_slide, win_size, temp_pwld, sws[0]);
        }
        //   auto after = std::chrono::duration<double, std::milli>(
        //     std::chrono::system_clock::now().time_since_epoch()
        // ).count();
        // std::cout<<"before - after"<<std::to_string(before - after)<<std::endl;
        // cut_voxel(surf_map, pvec_buf[win_count-1], win_count-1, surf_map_slide, win_size, pwld, sws[0]);
        //  // cut_voxel(temp_surf_map, pvec_buf[win_count-1], win_count-1, surf_map_slide, win_size, pwld, sws[0]);
        //  //cut_voxel(temp_surf_map, temp_pvec, win_count-1, surf_map_slide, win_size, temp_pwld, sws[0]);

        t2 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
        // // multi_recut(temp_surf_map, win_count, x_buf, voxhess, sws);
        multi_recut(surf_map_slide, win_count, x_buf, voxhess, sws);


        //静态点和滑窗点统计
        // std::cout<<"oxhess.frame_idxs.size():"<<voxhess.frame_idxs.size()<<std::endl;
        // int static_num = 0;
        // int slide_num = 0;
        // for(int i = 0; i < voxhess.frame_idxs.size(); i++)
        // {
        //   for(int j = 0; j < voxhess.frame_idxs[i].size(); j++)
        //   {
        //     if(voxhess.frame_idxs[i][j] == -1)
        //     {
        //       static_num++;

        //     }
        //     else{
        //       slide_num++;
        //     }

        //   }

        // }
        // std::cout<<"static_num:"<<static_num<<std::endl;
        // std::cout<<"slide_num:"<<slide_num<<std::endl;


        t3 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
        // std::cout<<"degrade_cnt:"<<degrade_cnt<<std::endl;
        if (degrade_cnt > degrade_bound)
        {
          degrade_cnt = 0;
          system_reset(imus);

          last_pos = x_curr.p;
          jour = 0;


          reset_flag = 1;

          motion_init_flag = 1;
          history_kfsize = 0;

          continue;
        }
      }

      if (win_count >= win_size)
      {
        t4 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();

        if (g_update == 2)
        {
          LI_BA_OptimizerGravity opt_lsv;
          vector<double> resis;
          opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, &hess, 5);
          printf("g update: %lf %lf %lf: %lf\n", x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm());
          g_update = 0;
          x_curr.g = x_buf[win_count - 1].g;
        }
        else
        {
          LI_BA_Optimizer opt_lsv;
          opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, &hess);
        }



        x_curr.R = x_buf[win_count - 1].R;
        x_curr.p = x_buf[win_count - 1].p;
        t5 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();

        ResultOutput::instance().pub_localmap(mgsize, sessionNames.size() - 1, pvec_buf, x_buf, pcl_path, win_base, win_count);

        multi_margi(surf_map_slide, jour, win_count, x_buf, voxhess, sws[0]);
        t6 = std::chrono::duration<double, std::milli>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();

        if ((win_base + win_count) % 10 == 0)
        {
          double spat = (x_curr.p - last_pos).norm();
          if (spat > 0.5)
          {
            jour += spat;
            last_pos = x_curr.p;
            release_flag = true;
          }
        }


        for (int i = 0; i < win_size; i++)
        {
          mp[i] += mgsize;
          if (mp[i] >= win_size)
            mp[i] -= win_size;
        }

        for (int i = mgsize; i < win_count; i++)
        {
          x_buf[i - mgsize] = x_buf[i];
          PVecPtr pvec_tem = pvec_buf[i - mgsize];
          pvec_buf[i - mgsize] = pvec_buf[i];
          pvec_buf[i] = pvec_tem;
        }

        for (int i = win_count - mgsize; i < win_count; i++)
        {
          x_buf.pop_back();
          pvec_buf.pop_back();

          delete imu_pre_buf.front();
          imu_pre_buf.pop_front();
        }

        win_base += mgsize;
        win_count -= mgsize;
      }

      double t_end = std::chrono::duration<double, std::milli>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
      double mem = get_memory();
      // printf("%d: %.4lf: %.4lf %.4lf %.4lf %.4lf %.4lf %.2lfGb %.1lf\n", win_base+win_count, t_end-t0, t1-t0, t2-t1, t3-t2, t5-t4, t6-t5, mem, jour);
      //  std::cout<<"t_end-t0:"<<std::to_string(t_end-t0)<<std::endl;
      //  std::cout<<"t1-t0:"<<std::to_string(t1-t0)<<std::endl;
      //  std::cout<<"t2-t1:"<<std::to_string(t2-t1)<<std::endl;
      //  std::cout<<"t3-t2:"<<std::to_string(t3-t2)<<std::endl;
      //  std::cout<<"t4-t3:"<<std::to_string(t4-t3)<<std::endl;
      //  std::cout<<"t5-t4:"<<std::to_string(t5-t4)<<std::endl;
      //  std::cout<<"t6-t5:"<<std::to_string(t6-t5)<<std::endl;

      // printf("%d: %lf %lf %lf\n", win_base + win_count, x_curr.p[0], x_curr.p[1], x_curr.p[2]);
    }

    vector<OctoTree *> octos;
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
      iter->second->tras_ptr(octos);
      iter->second->clear_slwd(sws[0]);
      delete iter->second;
    }

    for (int i = 0; i < octos.size(); i++)
      delete octos[i];
    octos.clear();

    for (int i = 0; i < sws[0].size(); i++)
      delete sws[0][i];
    sws[0].clear();
    malloc_trim(0);
  }


};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cmn_voxel");
  ros::NodeHandle n;

  pub_cmap = n.advertise<sensor_msgs::PointCloud2>("/map_cmap", 100);
  pub_pmap = n.advertise<sensor_msgs::PointCloud2>("/map_pmap", 100);
  pub_scan = n.advertise<sensor_msgs::PointCloud2>("/map_scan", 100);
  pub_init = n.advertise<sensor_msgs::PointCloud2>("/map_init", 100);
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_curr_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_prev_path = n.advertise<sensor_msgs::PointCloud2>("/map_true", 100);

  mp = new int[10];
  for (int i = 0; i < 10; i++)
    mp[i] = i;

  VOXEL_SLAM vs(n);
  // mp = new int[vs.win_size];
  // for (int i = 0; i < vs.win_size; i++)
  //   mp[i] = i;


  vs.thd_odometry_localmapping(n);


  ros::spin();
  return 0;
}
