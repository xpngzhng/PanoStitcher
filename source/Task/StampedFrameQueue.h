#pragma once

#include <opencv2/core/core.hpp>
#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>

struct StampedFrame
{
    StampedFrame(const cv::Mat& frame_ = cv::Mat(), 
        long long int timeStamp_ = -1, int num_ = 0, int den_ = 1) :
        frame(frame_), timeStamp(timeStamp_), num(num_), den(den_) {}
    cv::Mat frame;
    long long int timeStamp;
    int num, den;
};

class StampedFrameQueue
{
    enum {DEFAULT_QUEUE_SIZE = 24, MAX_QUEUE_SIZE = 1024};
public:
    StampedFrameQueue(int maxSize_ = DEFAULT_QUEUE_SIZE) : 
        maxSize(maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_)) {};
    bool push(const StampedFrame& item)
    {
        bool ret = true;
        std::lock_guard<std::mutex> lock(mtxQueue);
        if (queue.size() > maxSize - 1)
        {
            while (queue.size() > maxSize - 1)
            {
                queue.pop_back();
            }
            ret = false;
        }
        queue.push_front(item);
        return ret;
    }
    bool pull(StampedFrame& item)
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        if (queue.empty())
        {
            item = StampedFrame();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
    }
private:
    int maxSize;
    std::deque<StampedFrame> queue;
    std::mutex mtxQueue;
};

class CompleteStampedFrameQueue
{
    enum { DEFAULT_QUEUE_SIZE = 24, MAX_QUEUE_SIZE = 1024 };
public:
    CompleteStampedFrameQueue(int maxSize_ = DEFAULT_QUEUE_SIZE) :
        maxSize(maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_)), pass(0) {};
    bool push(const StampedFrame& item)
    {
        bool ret = true;
        {
            std::lock_guard<std::mutex> lock(mtxQueue);
            queue.push_front(item);
        }
        condNonEmpty.notify_one();
        return ret;
    }
    bool pull(StampedFrame& item)
    {
        std::unique_lock<std::mutex> lock(mtxQueue);
        if (queue.empty() && !pass)
        {
            condNonEmpty.wait(lock, [this]{return (!this->queue.empty()) || this->pass;});
        }
        if (pass)
        {
            item = StampedFrame();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
    }
    int size()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        return queue.size();
    }
    void stop()
    {
        pass = 1;
        condNonEmpty.notify_one();
    }
    void resume()
    {
        pass = 0;
    }
private:
    int maxSize;
    std::deque<StampedFrame> queue;
    std::mutex mtxQueue;
    std::condition_variable condNonEmpty;
    int pass;
};

class StampedFrameArray
{
public:
    StampedFrameArray(int size_ = 0)
    {
        size = size_;
        frames.resize(size);
        state.store(0);
    }
    void setFrame(const StampedFrame& frame_, int index_)
    {
        frames[index_] = frame_;
        state.fetch_add(1);
        if (state.load() == size)
            condAllSet.notify_one();
        std::unique_lock<std::mutex> lock(mtxAllGet);
        condAllGet.wait(lock, [this]{return this->state.load() == 0;});
    }
    void getFrames(std::vector<StampedFrame>& frames_)
    {
        {
            std::unique_lock<std::mutex> lock(mtxAllSet);
            condAllSet.wait(lock, [this]{return this->state.load() == size;});
        }
        frames_ = frames;
        state.store(0);
        condAllGet.notify_all();
    }
private:
    int size;
    std::vector<StampedFrame> frames;
    std::atomic<int> state;
    std::condition_variable condAllSet;
    std::mutex mtxAllSet;
    std::condition_variable condAllGet;
    std::mutex mtxAllGet;
};

// This queue limits its size.
// If it reaches its maximum size, the oldest item
// will be deleted before the newly coming item is inserted.
// If it is empty, it will return an empty item and a false flag,
// it will not wait until an item comes.
template <typename ItemType>
class RealTimeQueue
{
    enum { DEFAULT_QUEUE_SIZE = 16, MAX_QUEUE_SIZE = 64 };
public:
    RealTimeQueue(int maxSize_ = DEFAULT_QUEUE_SIZE) :
        maxSize(maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_)) {};
    void setMaxSize(int maxSize_)
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        maxSize = maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_);
    }
    bool push(const ItemType& item)
    {
        bool ret = true;
        std::lock_guard<std::mutex> lock(mtxQueue);
        if (queue.size() > maxSize - 1)
        {
            while (queue.size() > maxSize - 1)
            {
                queue.pop_back();
            }
        }
        queue.push_front(item);
        return ret;
    }
    bool pull(ItemType& item)
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        if (queue.empty())
        {
            item = ItemType();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
    }
    int size()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        return queue.size();
    }
private:
    int maxSize;
    std::deque<ItemType> queue;
    std::mutex mtxQueue;
};

// This queue is also realtime in that it has size limit, 
// and items pushed in the queue early will be erased before newly coming item to be pushed.
// But if it is empty, pull operation will wait until a new item comes.
template <typename ItemType>
class ForceWaitRealTimeQueue
{
    enum { DEFAULT_QUEUE_SIZE = 16, MAX_QUEUE_SIZE = 64 };
public:
    ForceWaitRealTimeQueue(int maxSize_ = DEFAULT_QUEUE_SIZE) :
        maxSize(maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_)),
        pass(0) {};
    void setMaxSize(int maxSize_)
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        maxSize = maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_);
    }
    bool push(const ItemType& item)
    {
        bool ret = true;
        {
            std::lock_guard<std::mutex> lock(mtxQueue);
            if (queue.size() > maxSize - 1)
            {
                while (queue.size() > maxSize - 1)
                {
                    queue.pop_back();
                }
            }
            queue.push_front(item);
        }
        condNonEmpty.notify_one();
        return ret;
    }
    bool pull(ItemType& item)
    {
        std::unique_lock<std::mutex> lock(mtxQueue);
        if (queue.empty() && !pass)
        {
            condNonEmpty.wait(lock, [this]{return (!this->queue.empty()) || this->pass; });
        }
        if (pass)
        {
            item = ItemType();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
        pass = 0;
    }
    int size()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        return queue.size();
    }
    void stop()
    {
        pass = 1;
        condNonEmpty.notify_one();
    }
    void resume()
    {
        pass = 0;
    }
private:
    int maxSize;
    std::deque<ItemType> queue;
    std::mutex mtxQueue;
    std::condition_variable condNonEmpty;
    int pass;
};

// This queue keeps all pushed items and will not delete them except pull operations.
// If the queue is empty, pull request will wait until a new item is pushed.
// This queue DOES NOT HAVE SIZE LIMIT!!!!!!
template<typename ItemType>
class CompleteQueue
{
public:
    CompleteQueue() :  pass(0) {};
    bool push(const ItemType& item)
    {
        bool ret = true;
        {
            std::lock_guard<std::mutex> lock(mtxQueue);
            queue.push_front(item);
        }
        condNonEmpty.notify_one();
        return ret;
    }
    bool pull(ItemType& item)
    {
        std::unique_lock<std::mutex> lock(mtxQueue);
        if (queue.empty() && !pass)
        {
            condNonEmpty.wait(lock, [this]{return (!this->queue.empty()) || this->pass; });
        }
        if (pass)
        {
            item = ItemType();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
        pass = 0;
    }
    int size()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        return queue.size();
    }
    void stop()
    {
        pass = 1;
        condNonEmpty.notify_one();
    }
    void resume()
    {
        pass = 0;
    }
private:
    std::deque<ItemType> queue;
    std::mutex mtxQueue;
    std::condition_variable condNonEmpty;
    int pass;
};

// This queue is a size-limited version of the above complete queue
template<typename ItemType>
class BoundedCompleteQueue
{
    enum { DEFAULT_QUEUE_SIZE = 16, MAX_QUEUE_SIZE = 64 };
public:
    BoundedCompleteQueue(int maxSize_ = DEFAULT_QUEUE_SIZE) :
        maxSize(maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_)), pass(0) {};
    void setMaxSize(int maxSize_)
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        maxSize = maxSize_ <= 0 ? DEFAULT_QUEUE_SIZE : (maxSize_ > MAX_QUEUE_SIZE ? MAX_QUEUE_SIZE : maxSize_);
    }
    bool push(const ItemType& item)
    {
        std::unique_lock<std::mutex> lock(mtxQueue);
        if (queue.size() == maxSize && !pass)
        {
            condNotFull.wait(lock, [this]{return (this->queue.size() < maxSize) || this->pass; });
        }
        if (pass)
            return false;
        queue.push_front(item);
        condNonEmpty.notify_one();
        return true;
    }
    bool pull(ItemType& item)
    {
        std::unique_lock<std::mutex> lock(mtxQueue);
        if (queue.empty() && !pass)
        {
            condNonEmpty.wait(lock, [this]{return (!this->queue.empty()) || this->pass; });
        }
        if (pass)
        {
            item = ItemType();
            return false;
        }
        item = queue.back();
        queue.pop_back();
        condNotFull.notify_one();
        return true;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        queue.clear();
        pass = 0;
    }
    int size()
    {
        std::lock_guard<std::mutex> lock(mtxQueue);
        return queue.size();
    }
    void stop()
    {
        pass = 1;
        condNonEmpty.notify_one();
        condNotFull.notify_one();
    }
    void resume()
    {
        pass = 0;
    }
private:
    int maxSize;
    std::deque<ItemType> queue;
    std::mutex mtxQueue;
    std::condition_variable condNonEmpty, condNotFull;
    int pass;
};