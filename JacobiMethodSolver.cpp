#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>


struct JacobiMethodSolverConfig {
  static constexpr int N = 320;
  static constexpr int a = 10e5;
  static constexpr double epsilon = 10e-8;
  static constexpr int D_x = 2;
  static constexpr int D_y = 2;
  static constexpr int D_z = 2;
  static constexpr int x_0 = -1;
  static constexpr int y_0 = -1;
  static constexpr int z_0 = -1;

  static constexpr double h_x = D_x / (double) (N - 1);
  static constexpr double h_y = D_y / (double) (N - 1);
  static constexpr double h_z = D_z / (double) (N - 1);

  static constexpr double squaredH_x = h_x * h_x;
  static constexpr double squaredH_y = h_y * h_y;
  static constexpr double squaredH_z = h_z * h_z;

  int procNum = 0;
  int procRank = 0;
};

double phi(double x, double y, double z) {
  return (x * x) + (y * y) + (z * z);
}

double ro(double x, double y, double z) {
  return 6 - JacobiMethodSolverConfig::a * phi(x, y, z);
}

double calculateX(int i) {
  return JacobiMethodSolverConfig::x_0 + (i * JacobiMethodSolverConfig::h_x);
}

double calculateY(int j) {
  return JacobiMethodSolverConfig::y_0 + (j * JacobiMethodSolverConfig::h_y);
}

double calculateZ(int k) {
  return JacobiMethodSolverConfig::z_0 + (k * JacobiMethodSolverConfig::h_z);
}

int idx(int i, int j, int k) {
  return i * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N + j * JacobiMethodSolverConfig::N + k;
}

void initializePhi(int layerHeight, std::vector<double> &currentLayer, JacobiMethodSolverConfig &deSolverConfig) {
  for (int i = 0; i < layerHeight + 2; i++) {
    int relativeZ = i + ((deSolverConfig.procRank * layerHeight) - 1);
    double z = calculateZ(relativeZ);

    for (int j = 0; j < JacobiMethodSolverConfig::N; j++) {
      double x = calculateX(j);

      for (int k = 0; k < JacobiMethodSolverConfig::N; k++) {
        double y = calculateY(k);

        if (k != 0 && k != JacobiMethodSolverConfig::N - 1 &&
            j != 0 && j != JacobiMethodSolverConfig::N - 1 &&
            z != JacobiMethodSolverConfig::z_0 && z != JacobiMethodSolverConfig::z_0 + JacobiMethodSolverConfig::D_z) {
          currentLayer[idx(i, j, k)] = 0;
        } else {
          currentLayer[idx(i, j, k)] = phi(x, y, z);
        }

      }
    }
  }
}

void printCube(double *A) {
  for (int i = 0; i < JacobiMethodSolverConfig::N; i++) {
    for (int j = 0; j < JacobiMethodSolverConfig::N; j++) {
      for (int k = 0; k < JacobiMethodSolverConfig::N; k++) {
        printf(" %7.4f", A[idx(i, j, k)]);
      }
      std::cout << ";";
    }
    std::cout << std::endl;
  }
}

double calculateDelta(std::vector<double> &omega) {
  auto deltaMax = DBL_MIN;
  double x, y, z;
  for (int i = 0; i < JacobiMethodSolverConfig::N; i++) {
    x = calculateX(i);
    for (int j = 0; j < JacobiMethodSolverConfig::N; j++) {
      y = calculateY(j);
      for (int k = 0; k < JacobiMethodSolverConfig::N; k++) {
        z = calculateZ(k);
        deltaMax = std::max(deltaMax, (double) std::abs(omega[idx(i, j, k)] - phi(x, y, z)));
      }
    }
  }

  return deltaMax;
}

double updateLayer(int relativeZCoordinate, int layerIdx, std::vector<double> &currentLayer,
                   std::vector<double> &currentLayerBuf) {
  int absoluteZCoordinate = relativeZCoordinate + layerIdx;
  auto deltaMax = DBL_MIN;
  double x, y, z;

  if (absoluteZCoordinate == 0 || absoluteZCoordinate == JacobiMethodSolverConfig::N - 1) {
    memcpy(currentLayerBuf.data() + layerIdx * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
           currentLayer.data() + layerIdx * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
           JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N * sizeof(double));
    deltaMax = 0;
  } else {
    z = calculateZ(absoluteZCoordinate);

    for (int i = 0; i < JacobiMethodSolverConfig::N; i++) {
      x = calculateX(i);
      for (int j = 0; j < JacobiMethodSolverConfig::N; j++) {
        y = calculateY(j);

        if (i == 0 || i == JacobiMethodSolverConfig::N - 1 || j == 0 || j == JacobiMethodSolverConfig::N - 1) {
          currentLayerBuf[idx(layerIdx, i, j)] = currentLayer[idx(layerIdx, i, j)];
        } else {
          currentLayerBuf[idx(layerIdx, i, j)] =
                  ((currentLayer[idx(layerIdx + 1, i, j)] + currentLayer[idx(layerIdx - 1, i, j)]) /
                   JacobiMethodSolverConfig::squaredH_z
                   +
                   (currentLayer[idx(layerIdx, i + 1, j)] + currentLayer[idx(layerIdx, i - 1, j)]) /
                   JacobiMethodSolverConfig::squaredH_x
                   +
                   (currentLayer[idx(layerIdx, i, j + 1)] + currentLayer[idx(layerIdx, i, j - 1)]) /
                   JacobiMethodSolverConfig::squaredH_y
                   -
                   ro(x, y, z)) /
                  (2 / JacobiMethodSolverConfig::squaredH_x + 2 / JacobiMethodSolverConfig::squaredH_y +
                   2 / JacobiMethodSolverConfig::squaredH_z +
                   JacobiMethodSolverConfig::a);

          if (std::abs(currentLayerBuf[idx(layerIdx, i, j)] - currentLayer[idx(layerIdx, i, j)]) > deltaMax) {
            deltaMax = currentLayerBuf[idx(layerIdx, i, j)] - currentLayer[idx(layerIdx, i, j)];
          }

        }
      }
    }
  }

  return deltaMax;
}

int main(int argc, char *argv[]) {
  JacobiMethodSolverConfig deSolverConfig = JacobiMethodSolverConfig();
  std::vector<double> omega;
  double deltaForTerminationCriterion = DBL_MAX;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &deSolverConfig.procNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &deSolverConfig.procRank);
  MPI_Request req[4];

  if ((long) JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N > INT32_MAX) {
    std::cerr << "Grid size N = " << JacobiMethodSolverConfig::N << " is too big." << std::endl;
    return 1;
  }

  if (JacobiMethodSolverConfig::N % deSolverConfig.procNum && deSolverConfig.procRank == 0) {
    std::cerr << "Grid size N = " << JacobiMethodSolverConfig::N << " should be divisible by the procNum = "
              << deSolverConfig.procNum << std::endl;
    return 1;
  }


  int layerSize = JacobiMethodSolverConfig::N / deSolverConfig.procNum;
  int layerZCoordinate = deSolverConfig.procRank * layerSize - 1;

  int extendedLayerSize = (layerSize + 2) * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N;
  std::vector<double> currentLayer = std::vector<double>(extendedLayerSize);
  std::vector<double> currentLayerBuf = std::vector<double>(extendedLayerSize);

  initializePhi(layerSize, currentLayer, deSolverConfig);

  double startTime = MPI_Wtime();
  do {
    double procMaxDelta = DBL_MIN;
    double tmpMaxDelta;

    if (deSolverConfig.procRank != 0) {
      MPI_Isend(currentLayerBuf.data() + JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
                JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
                MPI_DOUBLE,
                deSolverConfig.procRank - 1,
                888,
                MPI_COMM_WORLD,
                &req[1]);

      MPI_Irecv(currentLayerBuf.data(),
                JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
                MPI_DOUBLE,
                deSolverConfig.procRank - 1,
                888,
                MPI_COMM_WORLD,
                &req[0]);
    }

    if (deSolverConfig.procRank != deSolverConfig.procNum - 1) {
      MPI_Isend(currentLayerBuf.data() + JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N * layerSize,
                JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
                MPI_DOUBLE,
                deSolverConfig.procRank + 1,
                888,
                MPI_COMM_WORLD,
                &req[3]);

      MPI_Irecv(currentLayerBuf.data() + JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N * (layerSize + 1),
                JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
                MPI_DOUBLE,
                deSolverConfig.procRank + 1,
                888,
                MPI_COMM_WORLD,
                &req[2]);
    }

    for (int LayerIdx = 2; LayerIdx < layerSize; LayerIdx++) {
      tmpMaxDelta = updateLayer(layerZCoordinate, LayerIdx, currentLayer, currentLayerBuf);
      procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);
    }

    if (deSolverConfig.procRank != deSolverConfig.procNum - 1) {
      MPI_Wait(&req[2], MPI_STATUS_IGNORE);
      MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }

    if (deSolverConfig.procRank != 0) {
      MPI_Wait(&req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }

    tmpMaxDelta = updateLayer(layerZCoordinate, 1, currentLayer, currentLayerBuf);
    procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);

    tmpMaxDelta = updateLayer(layerZCoordinate, layerSize, currentLayer, currentLayerBuf);
    procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);

    currentLayer = currentLayerBuf;

    MPI_Allreduce(&procMaxDelta,
                  &deltaForTerminationCriterion,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    std::cout << deltaForTerminationCriterion << std::endl;
  } while (deltaForTerminationCriterion > JacobiMethodSolverConfig::epsilon);

  double endTime = MPI_Wtime();

  if (deSolverConfig.procRank == 0) {
    omega = std::vector<double>(
            JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N);
  }

  MPI_Gather(currentLayer.data() + JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
             layerSize * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
             MPI_DOUBLE,
             omega.data(),
             layerSize * JacobiMethodSolverConfig::N * JacobiMethodSolverConfig::N,
             MPI_DOUBLE,
             0,
             MPI_COMM_WORLD);

  if (deSolverConfig.procRank == 0) {
    std::cout << "Time taken: " << endTime - startTime << " s" << std::endl;
    std::cout << "Delta: " << calculateDelta(omega) << std::endl;
    std::cout << "Number of processes: " << deSolverConfig.procNum << std::endl;
  }
  MPI_Finalize();
  return 0;
}
