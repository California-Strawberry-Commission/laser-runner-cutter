#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Iteration and indexing order: logicalIndex 0 is the oldest retained
// element, logicalIndex size() - 1 is the newest.

template <typename T>
class RingBuffer {
 public:
  template <bool IsConst>
  class Iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const T*, T*>;
    using reference = std::conditional_t<IsConst, const T&, T&>;

    Iterator() = default;

    // Convert iterator to const_iterator.
    template <bool OtherConst,
              std::enable_if_t<IsConst && !OtherConst, int> = 0>
    Iterator(const Iterator<OtherConst>& other)
        : ring_{other.ring_}, logicalIndex_{other.logicalIndex_} {}

    reference operator*() const {
      return (*ring_)[static_cast<std::size_t>(logicalIndex_)];
    }
    pointer operator->() const { return &(**this); }
    reference operator[](difference_type offset) const {
      return *(*this + offset);
    }

    Iterator& operator++() {
      ++logicalIndex_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator before{*this};
      ++logicalIndex_;
      return before;
    }

    Iterator& operator--() {
      --logicalIndex_;
      return *this;
    }

    Iterator operator--(int) {
      Iterator before{*this};
      --logicalIndex_;
      return before;
    }

    Iterator& operator+=(difference_type offset) {
      logicalIndex_ += offset;
      return *this;
    }

    Iterator& operator-=(difference_type offset) {
      logicalIndex_ -= offset;
      return *this;
    }

    Iterator operator+(difference_type offset) const {
      return Iterator{ring_, logicalIndex_ + offset};
    }
    Iterator operator-(difference_type offset) const {
      return Iterator{ring_, logicalIndex_ - offset};
    }

    friend Iterator operator+(difference_type offset, const Iterator& it) {
      return it + offset;
    }

    difference_type operator-(const Iterator& other) const {
      assert(ring_ == other.ring_);
      return logicalIndex_ - other.logicalIndex_;
    }

    bool operator==(const Iterator& other) const {
      return ring_ == other.ring_ && logicalIndex_ == other.logicalIndex_;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }
    bool operator<(const Iterator& other) const {
      return logicalIndex_ < other.logicalIndex_;
    }
    bool operator>(const Iterator& other) const {
      return logicalIndex_ > other.logicalIndex_;
    }
    bool operator<=(const Iterator& other) const {
      return logicalIndex_ <= other.logicalIndex_;
    }
    bool operator>=(const Iterator& other) const {
      return logicalIndex_ >= other.logicalIndex_;
    }

   private:
    using ringPointer =
        std::conditional_t<IsConst, const RingBuffer*, RingBuffer*>;

    Iterator(ringPointer ring, difference_type logicalIndex)
        : ring_{ring}, logicalIndex_{logicalIndex} {}

    ringPointer ring_{nullptr};
    difference_type logicalIndex_{0};

    friend class RingBuffer;
    template <bool>
    friend class Iterator;
  };

  using value_type = T;
  using size_type = std::size_t;
  using reference = T&;
  using const_reference = const T&;
  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;

  explicit RingBuffer(size_type capacity) : capacity_{capacity} {
    if (capacity_ == 0) {
      throw std::invalid_argument("RingBuffer capacity must be positive");
    }
    storage_.reserve(capacity_);
  }

  // Append an element, overwriting the oldest one once size() == capacity().
  // Each eviction shifts every index down by one.
  void push_back(const T& value) {
    assert(size_ == storage_.size());
    if (storage_.size() < capacity_) {
      storage_.push_back(value);
      ++size_;
    } else {
      storage_[oldest_] = value;
      oldest_ = (oldest_ + 1) % capacity_;
    }
  }

  void push_back(T&& value) {
    assert(size_ == storage_.size());
    if (storage_.size() < capacity_) {
      storage_.push_back(std::move(value));
      ++size_;
    } else {
      storage_[oldest_] = std::move(value);
      oldest_ = (oldest_ + 1) % capacity_;
    }
  }

  // Remove all elements.
  void clear() {
    storage_.clear();
    oldest_ = 0;
    size_ = 0;
  }

  reference operator[](size_type logicalIndex) {
    return storage_[physicalIndex(logicalIndex)];
  }
  const_reference operator[](size_type logicalIndex) const {
    return storage_[physicalIndex(logicalIndex)];
  }

  reference at(size_type logicalIndex) {
    if (logicalIndex >= size_) {
      throw std::out_of_range("RingBuffer index out of range");
    }
    return (*this)[logicalIndex];
  }

  const_reference at(size_type logicalIndex) const {
    if (logicalIndex >= size_) {
      throw std::out_of_range("RingBuffer index out of range");
    }
    return (*this)[logicalIndex];
  }

  // Oldest element.
  reference front() {
    assert(!empty());
    return (*this)[0];
  }

  const_reference front() const {
    assert(!empty());
    return (*this)[0];
  }

  // Most recently added element.
  reference back() {
    assert(!empty());
    return (*this)[size_ - 1];
  }

  const_reference back() const {
    assert(!empty());
    return (*this)[size_ - 1];
  }

  size_type size() const { return size_; }
  size_type capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }
  bool full() const { return size_ == capacity_; }

  iterator begin() { return iterator{this, 0}; }
  iterator end() { return iterator{this, static_cast<difference_type>(size_)}; }
  const_iterator begin() const { return const_iterator{this, 0}; }
  const_iterator end() const {
    return const_iterator{this, static_cast<difference_type>(size_)};
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

 private:
  size_type physicalIndex(size_type logicalIndex) const {
    assert(logicalIndex < size_);
    return (oldest_ + logicalIndex) % capacity_;
  }

  std::vector<T> storage_;
  size_type capacity_;
  size_type oldest_{0};
  size_type size_{0};
};
