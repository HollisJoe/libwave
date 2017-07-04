/**
 * @file
 * @ingroup optimization
 */

#ifndef WAVE_OPTIMIZATION_FACTOR_GRAPH_COMPOSEDVALUE_HPP
#define WAVE_OPTIMIZATION_FACTOR_GRAPH_COMPOSEDVALUE_HPP

#include "wave/optimization/factor_graph/FactorValue.hpp"
#include "wave/optimization/factor_graph/impl/define_composed_value.hpp"

namespace wave {
/** @addtogroup optimization
 *  @{ */

template <typename Derived, template <typename...> class... V>
class ComposedValue;

template <template <typename, typename> class D,
          typename Scalar,
          typename Options,
          template <typename...> class... V>
class ComposedValue<D<Scalar, Options>, V...> {
    using Derived = D<Scalar, Options>;

 public:
    constexpr static std::array<int, sizeof...(V)> BlockSizes{{
      V<Scalar>::RowsAtCompileTime...}};
    constexpr static int Size = tmp::array_sum(BlockSizes);
    constexpr static std::array<int, sizeof...(V)> BlockIndices =
      tmp::cumulative_array(BlockSizes);
    using ComposedMatrix = FactorValue<Scalar, Options, Size>;
    using ValueTuple = std::tuple<V<Scalar, Options>...>;


    ComposedValue() : mat{ComposedMatrix::Zero()} {}

    explicit ComposedValue(V<Scalar, Options>... args) {
        this->initMatrix(tmp::make_index_sequence<sizeof...(V)>{},
                         std::move(args)...);
    }

    /** Initialize from pointer to raw array */
    explicit ComposedValue(Scalar *dataptr) : mat{dataptr} {}

    Scalar *data() noexcept {
        return this->mat.data();
    }

    // Arithmetic operators

    Derived &operator-=(const Derived &rhs) {
        this->mat -= rhs.mat;
        return static_cast<Derived &>(*this);
    }

    const Derived operator-(const Derived &rhs) const {
        return Derived{*static_cast<Derived const *>(this)} -= rhs;
    }

    template <int I>
    Eigen::Ref<typename std::tuple_element<I, ValueTuple>::type>
    block() noexcept {
        const auto i = BlockIndices[I];
        const auto size = BlockSizes[I];
        return this->mat.template segment<size>(i);
    }

 private:
    template <int... Is>
    void initMatrix(tmp::index_sequence<Is...>, V<Scalar, Options>... args) {
        auto loop = {(this->block<Is>() = std::move(args), 0)...};
        (void) loop;
    }

    ComposedMatrix mat;
};

/** Specialization for square option */
template <template <typename, typename> class D,
          typename Scalar,
          template <typename...> class... V>
class ComposedValue<D<Scalar, FactorValueOptions::Square>, V...> {
    using Derived = D<Scalar, FactorValueOptions::Square>;

 public:
    constexpr static std::array<int, sizeof...(V)> BlockSizes{{
      V<Scalar>::RowsAtCompileTime...}};
    constexpr static int Size = tmp::array_sum(BlockSizes);
    constexpr static std::array<int, sizeof...(V)> BlockIndices =
      tmp::cumulative_array(BlockSizes);
    using ComposedMatrix = Eigen::Matrix<Scalar, Size, Size>;
    using ValueTuple = std::tuple<V<Scalar, FactorValueOptions::Square>...>;

    ComposedValue() : mat{ComposedMatrix::Zero()} {}

    /** Initialize from pointer to raw array */
    explicit ComposedValue(Scalar *dataptr) : mat{dataptr} {}

    Scalar *data() noexcept {
        return this->mat.data();
    }

    /** Given a reference to a diagonal block of `mat`, determine the index */
    template <typename T>
    int indexFromRef(const Eigen::Ref<T> &ref) const {
        // Use low-level math for now
        const auto diff = ref.data() - this->mat.data();
        if (diff < 0 || diff >= this->mat.size()) {
            throw std::logic_error(
              "ComposedValue::indexFromRef: invalid reference");
        }
        return diff % Size;
    }

    template <int I>
    Eigen::Ref<typename std::tuple_element<I, ValueTuple>::type>
    block() noexcept {
        const auto i = BlockIndices[I];
        const auto size = BlockSizes[I];
        return this->mat.template block<size, size>(i, i);
    }

    template <int I, int J>
    Eigen::Block<ComposedMatrix,
                 BlockSizes[I], BlockSizes[J]>
    block() noexcept {
        const auto i = BlockIndices[I];
        const auto j = BlockIndices[J];
        const auto rows = BlockSizes[I];
        const auto cols = BlockSizes[J];
        return this->mat.template block<rows, cols>(i, j);
    }

 private:
    ComposedMatrix mat;
};

/** @} group optimization */
}  // namespace wave


#define WAVE_DEFINE_COMPOSED_VALUE(NAME, ATTRIBUTES) \
    WAVE_DEFINE_COMPOSED_VALUE_IMPL(NAME, ATTRIBUTES)

#endif  // WAVE_OPTIMIZATION_FACTOR_GRAPH_COMPOSEDVALUE_HPP
