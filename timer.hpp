#include <cstdio>
#include <sys/time.h>

class Timer {
    public:
        Timer() { reset(); }
        ~Timer() {}

        void reset() {
            _begin = (struct timeval){0};
            _stop  = (struct timeval){0};
        }

        void start() {
            gettimeofday(&_begin, NULL);
            _stop = _begin;
        }

        void interval_timing(const char *info = NULL) {
            struct timeval now;
            gettimeofday(&now, NULL);
            if (info) printf("%s", info);
            time_interval(_stop, now);
            _stop = now;
        }

        void timing(const char *info = NULL) {
            struct timeval now;
            gettimeofday(&now, NULL);
            if (info) printf("%s", info);
            time_interval(_begin, now);
        }

        void time_interval(struct timeval begin, struct timeval end) {
            double interval = 0.0f;
            if (end.tv_usec < begin.tv_usec) {
                interval = end.tv_usec + 1e6 - begin.tv_usec
                         + (end.tv_sec - 1 - begin.tv_sec) * 1e6;
            } else {
                interval = end.tv_usec - begin.tv_usec
                         + (end.tv_sec - begin.tv_sec) * 1e6;
            }
            printf(" time consumed: %.2fus\n", interval);
        }

    private:
        struct timeval _begin;
        struct timeval _stop;
};
