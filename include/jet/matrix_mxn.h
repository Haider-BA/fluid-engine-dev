// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_MXN_H_
#define INCLUDE_JET_MATRIX_MXN_H_

#include <jet/array2.h>
#include <jet/functors.h>
#include <jet/vector_n.h>

namespace jet {

// MARK: MatrixExpression

template <typename T, typename E>
class MatrixExpression {
 public:
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    const E& operator()() const;
};

template <typename T>
class MatrixConstant : public MatrixExpression<T, MatrixConstant<T>> {
 public:
    MatrixConstant(size_t m, size_t n, const T& c);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    size_t _m;
    size_t _n;
    T _c;
};

template <typename T>
class MatrixIdentity : public MatrixExpression<T, MatrixIdentity<T>> {
 public:
    MatrixIdentity(size_t m);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    size_t _m;
};

// MARK: MatrixUnaryOp

template <typename T, typename E, typename Op>
class MatrixUnaryOp : public MatrixExpression<T, MatrixUnaryOp<T, E, Op>> {
 public:
    MatrixUnaryOp(const E& u);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    Op _op;
};

template <typename T, typename E>
class MatrixDiagonal : public MatrixExpression<T, MatrixDiagonal<T, E>> {
 public:
    MatrixDiagonal(const E& u, bool isDiag);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    bool _isDiag;
};

template <typename T, typename E>
class MatrixTriangular : public MatrixExpression<T, MatrixTriangular<T, E>> {
 public:
    MatrixTriangular(const E& u, bool isUpper, bool isStrict);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    bool _isUpper;
    bool _isStrict;
};

// MARK: MatrixUnaryOp Aliases

template <typename T, typename E, typename U>
using MatrixTypeCast = MatrixUnaryOp<T, E, TypeCast<U, T>>;

// MARK: MatrixBinaryOp

template <typename T, typename E1, typename E2, typename Op>
class MatrixBinaryOp
    : public MatrixExpression<T, MatrixBinaryOp<T, E1, E2, Op>> {
 public:
    MatrixBinaryOp(const E1& u, const E2& v);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E1& _u;
    const E2& _v;
    Op _op;
};

template <typename T, typename E, typename Op>
class MatrixScalarBinaryOp
    : public MatrixExpression<T, MatrixScalarBinaryOp<T, E, Op>> {
 public:
    MatrixScalarBinaryOp(const E& u, const T& v);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    T _v;
    Op _op;
};

//!
//! \tparam T   Element value type.
//! \tparam ME  Matrix expression.
//! \tparam VE  Vector expression.
//!
template <typename T, typename ME, typename VE>
class MatrixVectorMul : public VectorExpression<T, MatrixVectorMul<T, ME, VE>> {
 public:
    MatrixVectorMul(const ME& m, const VE& v);
    size_t size() const;
    T operator[](size_t i) const;

 private:
    const ME& _m;
    const VE& _v;
};

template <typename T, typename E1, typename E2>
class MatrixMul : public MatrixExpression<T, MatrixMul<T, E1, E2>> {
 public:
    MatrixMul(const E1& u, const E2& v);
    Size2 size() const;
    size_t rows() const;
    size_t cols() const;
    T operator()(size_t i, size_t j) const;

 private:
    const E1& _u;
    const E2& _v;
};

// MARK: MatrixBinaryOp Aliases

template <typename T, typename E1, typename E2>
using MatrixAdd = MatrixBinaryOp<T, E1, E2, std::plus<T>>;

template <typename T, typename E>
using MatrixScalarAdd = MatrixScalarBinaryOp<T, E, std::plus<T>>;

template <typename T, typename E1, typename E2>
using MatrixSub = MatrixBinaryOp<T, E1, E2, std::minus<T>>;

template <typename T, typename E>
using MatrixScalarSub = MatrixScalarBinaryOp<T, E, std::minus<T>>;

template <typename T, typename E>
using MatrixScalarRSub = MatrixScalarBinaryOp<T, E, RMinus<T>>;

template <typename T, typename E>
using MatrixScalarMul = MatrixScalarBinaryOp<T, E, std::multiplies<T>>;

template <typename T, typename E>
using MatrixScalarDiv = MatrixScalarBinaryOp<T, E, std::divides<T>>;

template <typename T, typename E>
using MatrixScalarRDiv = MatrixScalarBinaryOp<T, E, RDivides<T>>;

// MARK: MatrixMxN

//!
//! \brief M x N matrix class.
//!
//! This class defines M x N row-major matrix.
//!
//! \tparam T Type of the element.
//!
template <typename T>
class MatrixMxN final : public MatrixExpression<T, MatrixMxN<T>> {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "MatrixMxN only can be instantiated with floating point types");

    typedef Array2<T> ContainerType;
    typedef typename ContainerType::Iterator Iterator;
    typedef typename ContainerType::ConstIterator ConstIterator;

    // MARK: Constructors

    //! Constructs an empty matrix.
    MatrixMxN();

    //! Constructs m x n constant value matrix.
    MatrixMxN(size_t m, size_t n, const T& s = T(0));

    //!
    //! \brief Constructs a matrix with given initializer list \p lst.
    //!
    //! This constructor will build a matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! MatrixMxN<float> mat = {
    //!     {1.f, 2.f, 4.f, 3.f},
    //!     {9.f, 3.f, 5.f, 1.f},
    //!     {4.f, 8.f, 1.f, 5.f}
    //! };
    //! \endcode
    //!
    //! Note the initializer has 4x3 structure which will create 4x3 matrix.
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    MatrixMxN(const std::initializer_list<std::initializer_list<T>>& lst);

    //! Constructs a matrix with expression template.
    template <typename E>
    MatrixMxN(const MatrixExpression<T, E>& other);

    //! Constructs a m x n matrix with input array.
    //! \warning Ordering of the input elements is row-major.
    MatrixMxN(size_t m, size_t n, const T* arr);

    //! Copy constructor.
    MatrixMxN(const MatrixMxN& other);

    //! Move constructor.
    MatrixMxN(MatrixMxN&& other);

    // MARK: Basic setters

    //! Resizes to m x n matrix with initial value \p s.
    void resize(size_t m, size_t n, const T& s = T(0));

    //! Sets whole matrix with input scalar.
    void set(const T& s);

    //!
    //! \brief Sets a matrix with given initializer list \p lst.
    //!
    //! This function will fill the matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! MatrixMxN<float> mat;
    //! mat.set({
    //!     {1.f, 2.f, 4.f, 3.f},
    //!     {9.f, 3.f, 5.f, 1.f},
    //!     {4.f, 8.f, 1.f, 5.f}
    //! });
    //! \endcode
    //!
    //! Note the initializer has 4x3 structure which will resize to 4x3 matrix.
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    void set(const std::initializer_list<std::initializer_list<T>>& lst);

    //! Copies from input matrix expression.
    template <typename E>
    void set(const MatrixExpression<T, E>& other);

    //! Copies from input array.
    //! \warning Ordering of the input elements is row-major.
    void set(size_t m, size_t n, const T* arr);

    //! Sets diagonal elements with input scalar.
    void setDiagonal(const T& s);

    //! Sets off-diagonal elements with input scalar.
    void setOffDiagonal(const T& s);

    //! Sets i-th row with input vector.
    template <typename E>
    void setRow(size_t i, const VectorExpression<T, E>& row);

    //! Sets j-th column with input vector.
    template <typename E>
    void setColumn(size_t j, const VectorExpression<T, E>& col);

    // MARK: Basic getters
    template <typename E>
    bool isEqual(const MatrixExpression<T, E>& other) const;

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    template <typename E>
    bool isSimilar(const MatrixExpression<T, E>& other,
                   double tol = std::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    bool isSquare() const;

    //! Returns the size of this matrix.
    Size2 size() const;

    //! Returns number of rows of this matrix.
    size_t rows() const;

    //! Returns number of columns of this matrix.
    size_t cols() const;

    //! Returns data pointer of this matrix.
    T* data();

    //! Returns constant pointer of this matrix.
    const T* const data() const;

    //! Returns the begin iterator of the matrix.
    Iterator begin();

    //! Returns the begin const iterator of the matrix.
    ConstIterator begin() const;

    //! Returns the end iterator of the matrix.
    Iterator end();

    //! Returns the end const iterator of the matrix.
    ConstIterator end() const;

    // MARK: Binary operator methods - new instance = this instance (+) input

    //! Returns this matrix + input scalar.
    MatrixScalarAdd<T, MatrixMxN> add(const T& s) const;

    //! Returns this matrix + input matrix (element-wise).
    template <typename E>
    MatrixAdd<T, MatrixMxN, E> add(const E& m) const;

    //! Returns this matrix - input scalar.
    MatrixScalarSub<T, MatrixMxN> sub(const T& s) const;

    //! Returns this matrix - input matrix (element-wise).
    template <typename E>
    MatrixSub<T, MatrixMxN, E> sub(const E& m) const;

    //! Returns this matrix * input scalar.
    MatrixScalarMul<T, MatrixMxN> mul(const T& s) const;

    //! Returns this matrix * input vector.
    template <typename VE>
    MatrixVectorMul<T, MatrixMxN, VE> mul(
        const VectorExpression<T, VE>& v) const;

    //! Returns this matrix * input matrix.
    template <typename E>
    MatrixMul<T, MatrixMxN, E> mul(const E& m) const;

    //! Returns this matrix / input scalar.
    MatrixScalarDiv<T, MatrixMxN> div(const T& s) const;

    // MARK: Binary operator methods - new instance = input (+) this instance
    //! Returns input scalar + this matrix.
    MatrixScalarAdd<T, MatrixMxN> radd(const T& s) const;

    //! Returns input matrix + this matrix (element-wise).
    template <typename E>
    MatrixAdd<T, MatrixMxN, E> radd(const E& m) const;

    //! Returns input scalar - this matrix.
    MatrixScalarRSub<T, MatrixMxN> rsub(const T& s) const;

    //! Returns input matrix - this matrix (element-wise).
    template <typename E>
    MatrixSub<T, MatrixMxN, E> rsub(const E& m) const;

    //! Returns input scalar * this matrix.
    MatrixScalarMul<T, MatrixMxN> rmul(const T& s) const;

    //! Returns input matrix * this matrix.
    template <typename E>
    MatrixMul<T, MatrixMxN, E> rmul(const E& m) const;

    //! Returns input matrix / this scalar.
    MatrixScalarRDiv<T, MatrixMxN> rdiv(const T& s) const;

    // MARK: Augmented operator methods - this instance (+)= input

    //! Adds input scalar to this matrix.
    void iadd(const T& s);

    //! Adds input matrix to this matrix (element-wise).
    template <typename E>
    void iadd(const E& m);

    //! Subtracts input scalar from this matrix.
    void isub(const T& s);

    //! Subtracts input matrix from this matrix (element-wise).
    template <typename E>
    void isub(const E& m);

    //! Multiplies input scalar to this matrix.
    void imul(const T& s);

    //! Multiplies input matrix to this matrix.
    template <typename E>
    void imul(const E& m);

    //! Divides this matrix with input scalar.
    void idiv(const T& s);

    // MARK: Modifiers

    //! Transposes this matrix.
    void transpose();

    //!
    //! \brief Inverts this matrix.
    //!
    //! This function computes the inverse using Gaussian elimination method.
    //!
    void invert();

    // MARK: Complex getters
    //! Returns sum of all elements.
    T sum() const;

    //! Returns average of all elements.
    T avg() const;

    //! Returns minimum among all elements.
    T min() const;

    //! Returns maximum among all elements.
    T max() const;

    //! Returns absolute minimum among all elements.
    T absmin() const;

    //! Returns absolute maximum among all elements.
    T absmax() const;

    //! Returns sum of all diagonal elements.
    //! \warning Should be a square matrix.
    T trace() const;

    //! Returns determinant of this matrix.
    T determinant() const;

    //! Returns diagonal part of this matrix.
    MatrixDiagonal<T, MatrixMxN> diagonal() const;

    //! Returns off-diagonal part of this matrix.
    MatrixDiagonal<T, MatrixMxN> offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    MatrixTriangular<T, MatrixMxN> strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    MatrixTriangular<T, MatrixMxN> strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    MatrixTriangular<T, MatrixMxN> lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    MatrixTriangular<T, MatrixMxN> upperTri() const;

    //! Returns transposed matrix.
    MatrixMxN transposed() const;

    //! Returns inverse matrix.
    MatrixMxN inverse() const;

    template <typename U>
    MatrixTypeCast<U, MatrixMxN, T> castTo() const;

    // MARK: Setter operators

    //! Assigns input matrix.
    template <typename E>
    MatrixMxN& operator=(const E& m);

    //! Copies to this matrix.
    MatrixMxN& operator=(const MatrixMxN& other);

    //! Moves to this matrix.
    MatrixMxN& operator=(MatrixMxN&& other);

    //! Addition assignment with input scalar.
    MatrixMxN& operator+=(const T& s);

    //! Addition assignment with input matrix (element-wise).
    template <typename E>
    MatrixMxN& operator+=(const E& m);

    //! Subtraction assignment with input scalar.
    MatrixMxN& operator-=(const T& s);

    //! Subtraction assignment with input matrix (element-wise).
    template <typename E>
    MatrixMxN& operator-=(const E& m);

    //! Multiplication assignment with input scalar.
    MatrixMxN& operator*=(const T& s);

    //! Multiplication assignment with input matrix.
    template <typename E>
    MatrixMxN& operator*=(const E& m);

    //! Division assignment with input scalar.
    MatrixMxN& operator/=(const T& s);

    // MARK: Getter operators

    //! Returns reference of i-th element.
    T& operator[](size_t i);

    //! Returns constant reference of i-th element.
    const T& operator[](size_t i) const;

    //! Returns reference of (i,j) element.
    T& operator()(size_t i, size_t j);

    //! Returns constant reference of (i,j) element.
    const T& operator()(size_t i, size_t j) const;

    //! Returns true if is equal to m.
    template <typename E>
    bool operator==(const MatrixExpression<T, E>& m) const;

    //! Returns true if is not equal to m.
    template <typename E>
    bool operator!=(const MatrixExpression<T, E>& m) const;

    // MARK: Helpers

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func. The callback function takes matrix's element as its
    //! input. The order of execution will be the same as the nested for-loop
    //! below:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! for (size_t i = 0; i < mat.rows(); ++i) {
    //!     for (size_t j = 0; j < mat.cols(); ++j) {
    //!         func(mat(i, j));
    //!     }
    //! }
    //! \endcode
    //!
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.forEach([](double elem) {
    //!     printf("%d\n", elem);
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEach(Callback func) const;

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func. The callback function takes two parameters which are
    //! the (i, j) indices of the matrix. The order of execution will be the
    //! same as the nested for-loop below:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! for (size_t i = 0; i < mat.rows(); ++i) {
    //!     for (size_t j = 0; j < mat.cols(); ++j) {
    //!         func(i, j);
    //!     }
    //! }
    //! \endcode
    //!
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.forEachIndex([&](size_t i, size_t j) {
    //!     mat(i, j) = 4.0 * i + 7.0 * j + 1.5;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEachIndex(Callback func) const;

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index in
    //!     parallel.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func. The callback function takes matrix's element as its
    //! input. The order of execution will be non-deterministic since it runs in
    //! parallel. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.parallelForEach([](double& elem) {
    //!     elem *= 2.0;
    //! });
    //! \endcode
    //!
    //! The parameter type of the callback function doesn't have to be T&, but
    //! const T& or T can be used as well.
    //!
    template <typename Callback>
    void parallelForEach(Callback func);

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index in
    //!     parallel using multi-threading.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func in parallel using multi-threading. The callback
    //! function takes two parameters which are the (i, j) indices of the
    //! matrix. The order of execution will be non-deterministic since it runs
    //! in parallel. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.parallelForEachIndex([&](size_t i, size_t j) {
    //!     mat(i, j) *= 2;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void parallelForEachIndex(Callback func) const;

    // MARK: Builders

    //! Makes a m x n matrix with zeros.
    static MatrixConstant<T> makeZero(size_t m, size_t n);

    //! Makes a m x m matrix with all diagonal elements to 1, and other elements
    //! to 0.
    static MatrixIdentity<T> makeIdentity(size_t m);

 private:
    ContainerType _elements;
};

//! Float-type M x N matrix.
typedef MatrixMxN<float> MatrixMxNF;

//! Double-type M x N matrix.
typedef MatrixMxN<double> MatrixMxND;

}  // namespace jet

#include "detail/matrix_mxn-inl.h"

#endif  // INCLUDE_JET_MATRIX_MXN_H_
