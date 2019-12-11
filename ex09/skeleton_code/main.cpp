#include <cassert>
#include <map>
#include <mpi.h>
#include <iostream>

#include "def.h"
#include "index.h"
#include "op.h"
#include "io.h"

// Adds vectors.
Vect Add(const Vect& a, const Vect& b) {
  auto r = a;
  for (Size i = 0; i < a.v.size(); ++i) {
    r.v[i] += b.v[i];
  }
  return r;
}

// Multiplies vector and scalar.
Vect Mul(const Vect& a, Real k) {
  auto r = a;
  for (auto& e : r.v) {
    e *= k;
  }
  return r;
}

// Multiplies matrix and vector.
Vect Mul(const Matr& a, const Vect& u, MPI_Comm comm) {
  /*
  a: local matrix patch
  v: local vector patch
  comm: communicator
  */

  // TODO 2a
  // get the current rank and the total number of ranks
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);
  Vect v;   // u(t+1)
  std::vector<Size> col; // indices (global column) requiring communication 
  std::vector<Size> row; // row of indices requiring communication
  std::vector<Size> rank; // rank of process for request

  // loop over rows L = Nx* Ny/ (Nbx* Nby)
  for(Size i = 0; i < L; ++i) {
    Real temp = 0.0; // store matrix-row vector product
    // loop over columns
    for(Size k= a.ki[i]; k< a.ki[i+1]; ++k){
      Size rc = GlbToRank(a.gjk[k]); // get rank handling u-vector entries of current column
      // check if index is stored in memory of local rank or if communication is needed
      if(rc != r){
        col.push_back(a.gjk[k]); // store global indices
        row.push_back(i);       // store row
        rank.push_back(rc);     // store rank
      } else {
        temp+= a.a[k]* u.v[GlbToLoc(a.gjk[k])]; // matrix-row vector product
      }
    }
    v.v.push_back(temp); // store solution in vector
  }

  // loop over ranks
  int* count_rank [nr];
  for (int i = 0; i< rank.size(); ++i) {
    count_rank[rank[i]]++;
  }

  // use MPI_Allreduce() to compute the number of messages that every processor needs to receive
  std::vector<Size> vect_count (nr); // count of requests for all ranks
  MPI_Allreduce(&count_rank, &vect_count, nr, MPI_UNSIGNED_LONG, MPI_SUM, comm);
  Size no_msg = vect_count[r]; // get (requests for current rank

  // send and receive the indices of the columns which need
  // to be communicated to the other processes
  int tag = 42;
  std::vector<MPI_Request> send_requ_col(col.size(), MPI_REQUEST_NULL); // send requests for own missing values
  std::vector<MPI_Status> send_stat_col(col.size());
  std::vector<MPI_Request> recv_requ_col(no_msg, MPI_REQUEST_NULL); // receive requests for missing values of other processes
  std::vector<MPI_Status> recv_stat_col(no_msg);

  std::vector<Size> col_comm (no_msg); // vector to store column indices for communication

  // send requests (column number) to other proceses
  for(Size i = 0; i<col.size(); ++i){
    MPI_Isend(&col[i], 1, MPI_INT, rank[i], tag, comm, &send_requ_col[i]);
  }

  // receive requests -> get information of who was the sender from recv_stat_col
  for(int i = 0; i<no_msg; ++i){
    MPI_Irecv(&col_comm[i], 1, MPI_INT, MPI_ANY_SOURCE, tag, comm, &recv_requ_col[i]);
  }

  // wait for all sends and receives to finish before conitnuing
  MPI_Waitall(send_requ_col.size(), send_requ_col.data(), send_stat_col.data());
  MPI_Waitall(recv_requ_col.size(), recv_requ_col.data(), recv_stat_col.data());

  // send and receive the requested elements of vector u
  std::vector<MPI_Request> send_requ_val(col.size(), MPI_REQUEST_NULL); // send the requested values for other processes
  std::vector<MPI_Status> send_stat_val(col.size());
  std::vector<MPI_Request> recv_requ_val(no_msg, MPI_REQUEST_NULL); // receive the own requested values
  std::vector<MPI_Status> recv_stat_val(no_msg);

  // vector to receive values
  std::vector<double> u_missing (no_msg);

  // send requested values; get source from MPI_STATUS; use column number as tag
  for(int i = 0; i<no_msg; ++i){
    MPI_Isend(&u.v[col_comm[i]], 1, MPI_DOUBLE, send_stat_col[i].MPI_SOURCE, col_comm[i], comm, &send_requ_val[i]);
  }

  // receive requested values; Source rank; tag column
  for(Size i = 0; i<col.size(); ++i){
    MPI_Irecv(&u_missing[i], 1, MPI_DOUBLE, rank[i], col[i], comm, &recv_requ_val[i]);
  }

  // wait for all sends and receives to finish before continuing
  MPI_Waitall(send_requ_val.size(), send_requ_val.data(), send_stat_val.data());
  MPI_Waitall(recv_requ_val.size(), recv_requ_val.data(), recv_stat_val.data());

  // multiply received data (i) with vector (column) and add to solution (row)
  for(Size i = 0; i<col.size(); ++i){
    v.v[row[i]]+= a.a[col[i]]* u_missing[i];
  }

  return v;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  // Laplacian -> for each process own patch?
  Matr a = GetLapl(r);

  // Initial vector
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0]) / NX;
    Real y = Real(xy[1]) / NY;
    Real dx = x - 0.5;
    Real dy = y - 0.5;
    Real r = 0.2;
    u.v.push_back(dx*dx + dy*dy < r*r ? 1. : 0.);
  }

  Write(u, comm, "u0");

  const Size nt = 10; // number of time steps
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(a, u, comm);

    Real k = 0.25; // scaling, k <= 0.25 required for stability.
    du = Mul(du, k);
    u = Add(u, du);
  }

  Write(u, comm, "u1");

  MPI_Finalize();
}
