/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace kbd {
  // https://stackoverflow.com/a/25389481/11927397
  template <class Derived>
  void write_binary(const std::string& filename,
                    const Eigen::PlainObjectBase<Derived>& matrix) {
    std::ofstream out(filename,
                      std::ios::out | std::ios::binary | std::ios::trunc);
    if (out.is_open()) {
      typedef typename Derived::Index Index;
      typedef typename Derived::Scalar Scalar;
      Index rows = matrix.rows(), cols = matrix.cols();
      out.write(reinterpret_cast<char*>(&rows),
                sizeof(typename Derived::Index));
      out.write(reinterpret_cast<char*>(&cols),
                sizeof(typename Derived::Index));
      out.write(reinterpret_cast<const char*>(matrix.data()),
                rows * cols *
                    static_cast<typename Derived::Index>(
                        sizeof(typename Derived::Scalar)));
      out.close();
    } else {
      std::cout << "Can not write to file: " << filename << std::endl;
    }
  }

  template <class Derived>
  void read_binary(const std::string& filename,
                   Eigen::PlainObjectBase<Derived>& matrix) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open()) {
      typename Derived::Index rows = 0, cols = 0;
      in.read(reinterpret_cast<char*>(&rows), sizeof(typename Derived::Index));
      in.read(reinterpret_cast<char*>(&cols), sizeof(typename Derived::Index));
      matrix.resize(rows, cols);
      in.read(reinterpret_cast<char*>(matrix.data()),
              rows * cols *
                  static_cast<typename Derived::Index>(
                      sizeof(typename Derived::Scalar)));
      in.close();
    } else {
      std::cout << "Can not open binary matrix file: " << filename << std::endl;
    }
  }

  // https://scicomp.stackexchange.com/a/21438
  template <class SparseMatrix>
  void write_binary_sparse(const std::string& filename,
                           const SparseMatrix& matrix) {
    assert(matrix.isCompressed() == true);
    std::ofstream out(filename,
                      std::ios::binary | std::ios::out | std::ios::trunc);
    if (out.is_open()) {
      typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
      rows = matrix.rows();
      cols = matrix.cols();
      nnzs = matrix.nonZeros();
      outS = matrix.outerSize();
      innS = matrix.innerSize();

      out.write(reinterpret_cast<char*>(&rows),
                sizeof(typename SparseMatrix::Index));
      out.write(reinterpret_cast<char*>(&cols),
                sizeof(typename SparseMatrix::Index));
      out.write(reinterpret_cast<char*>(&nnzs),
                sizeof(typename SparseMatrix::Index));
      out.write(reinterpret_cast<char*>(&outS),
                sizeof(typename SparseMatrix::Index));
      out.write(reinterpret_cast<char*>(&innS),
                sizeof(typename SparseMatrix::Index));

      typename SparseMatrix::Index sizeIndexS =
          static_cast<typename SparseMatrix::Index>(
              sizeof(typename SparseMatrix::StorageIndex));
      typename SparseMatrix::Index sizeScalar =
          static_cast<typename SparseMatrix::Index>(
              sizeof(typename SparseMatrix::Scalar));
      out.write(reinterpret_cast<const char*>(matrix.valuePtr()),
                sizeScalar * nnzs);
      out.write(reinterpret_cast<const char*>(matrix.outerIndexPtr()),
                sizeIndexS * outS);
      out.write(reinterpret_cast<const char*>(matrix.innerIndexPtr()),
                sizeIndexS * nnzs);

      out.close();
    } else {
      std::cout << "Can not write to file: " << filename << std::endl;
    }
  }

  template <class SparseMatrix>
  void read_binary_sparse(const std::string& filename, SparseMatrix& matrix) {
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    if (in.is_open()) {
      typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
      typename SparseMatrix::Index sizeScalar =
          static_cast<typename SparseMatrix::Index>(
              sizeof(typename SparseMatrix::Scalar));
      typename SparseMatrix::Index sizeIndex =
          static_cast<typename SparseMatrix::Index>(
              sizeof(typename SparseMatrix::Index));
      typename SparseMatrix::Index sizeIndexS =
          static_cast<typename SparseMatrix::Index>(
              sizeof(typename SparseMatrix::StorageIndex));
      std::cout << sizeScalar << " " << sizeIndex << std::endl;
      in.read(reinterpret_cast<char*>(&rows), sizeIndex);
      in.read(reinterpret_cast<char*>(&cols), sizeIndex);
      in.read(reinterpret_cast<char*>(&nnz), sizeIndex);
      in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
      in.read(reinterpret_cast<char*>(&inSz), sizeIndex);

      matrix.resize(rows, cols);
      matrix.makeCompressed();
      matrix.resizeNonZeros(nnz);

      in.read(reinterpret_cast<char*>(matrix.valuePtr()), sizeScalar * nnz);
      in.read(reinterpret_cast<char*>(matrix.outerIndexPtr()),
              sizeIndexS * outSz);
      in.read(reinterpret_cast<char*>(matrix.innerIndexPtr()),
              sizeIndexS * nnz);

      matrix.finalize();
      in.close();
    } // file is open
    else {
      std::cout << "Can not open binary sparse matrix file: " << filename
                << std::endl;
    }
  }
} // namespace kbd