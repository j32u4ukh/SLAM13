### Eigen::LDLT< _MatrixType, _UpLo >

```c++
// 解方程
// 我們求解 matrix * x = v 這個方程(求解 x)
// N的大小在前邊的宏里定義，矩陣數值由隨機數生成
// 對於正定矩陣，可以用 cholesky 分解來解方程
matrix.ldlt().solve(v);
```

Robust Cholesky decomposition of a matrix with pivoting. 

### Eigen::aligned_allocator

STL compatible allocator to use with types requiring a non standrad alignment. 

The memory is aligned as for dynamically aligned matrix/array types such as MatrixXd. By default, it will thus provide at least 16 bytes alignment and more in following cases:

* 32 bytes alignment if AVX is enabled.
* 64 bytes alignment if AVX512 is enabled.

> 應該是用於分配記憶體空間，bytes 對齊應該是可以使得數據查找更有效率

### EIGEN_MAKE_ALIGNED_OPERATOR_NEW

[參考](https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html)

If you define a structure having members of **fixed-size vectorizable** Eigen types, you must overload its "operator new" so that it generates 16-bytes-aligned pointers. Fortunately, Eigen provides you with a macro **EIGEN_MAKE_ALIGNED_OPERATOR_NEW** that does that for you.

```c++
class Foo
{
//   ...
  Eigen::Vector2d v;
//   ...
};
 
// ...
 
Foo *foo = new Foo;
```

When you have a class Foo like above, and you dynamically allocate a new Foo as above, then, since Foo doesn't have aligned "operator new", the returned pointer foo is not necessarily 128-bit aligned.

**The alignment attribute of the member v is then relative to the start of the class, foo.** If the foo pointer wasn't aligned, then foo->v won't be aligned either!

The solution is to let class Foo have an aligned "operator new", as we showed in the previous section.

#### What about dynamic-size matrices and vectors?

Dynamic-size matrices and vectors, such as Eigen::VectorXd, allocate dynamically their own array of coefficients, so they take care of requiring absolute alignment automatically. 

So they don't cause this issue. The issue discussed here is only with fixed-size vectorizable matrices and vectors.
