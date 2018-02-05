#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include <numa.h>
#include <omp.h>

#include <timer.hpp>

///////////////////////////////////////////////////////////////////////////

void read_data(const int size,
               double *_voxels_real,
               double *_voxels_imag,
               double *_voxels_weit)
{
    FILE *fp;

    if((fp = fopen("real.dat", "wb")) != NULL)
        assert(size == fread(_voxels_real, sizeof(double), size, fp));
    fclose(fp);

    if((fp = fopen("imag.dat", "wb")) != NULL)
        assert(fread(_voxels_imag, sizeof(double), size, fp));
    fclose(fp);

    if((fp = fopen("weit.dat", "wb")) != NULL)
        assert(fread(_voxels_weit, sizeof(double), size, fp));
    fclose(fp);
}

void numa_oblivious_test()
{
    const int dim = 280;
    const int batchsize = 29093774;

    Timer timer;
    timer.start();

    double *_volume = (double*)malloc(dim * dim * (dim / 2 + 1) * 2 * sizeof(double));
    double *_weight = (double*)malloc(dim * dim * (dim / 2 + 1) * sizeof(double));

    int *_coords = (int*)malloc(batchsize * sizeof(int));
    double *_voxels_real = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_imag = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_weit = (double*)malloc(batchsize * sizeof(double));

    timer.interval_timing("Data reading");

    int pos = 0;

    #pragma omp parallel for
    for (int n = 0; n < batchsize; ++n) {
        int current = pos + n;

        int index = _coords[current];

        #pragma omp atomic
        _volume[index * 2] += _voxels_real[current];
        #pragma omp atomic
        _volume[index * 2 + 1] += _voxels_imag[current];
        #pragma omp atomic
        _weight[index * 2] += _voxels_weit[current];
    }

    timer.interval_timing("Accumulating");

    free(_volume);
    free(_weight);
    free(_coords);
    free(_voxels_real);
    free(_voxels_imag);
    free(_voxels_weit);
}

void numa_aware_test()
{
    if (numa_available() < 0)  {
        printf("Error: no numa node avaliable on this system...\n");
        exit(1);
    }

    /* detect cpus and numa nodes */
    int ncpus = numa_num_configured_cpus();
    int nodes = numa_num_configured_nodes();
    int cpus_per_node = ncpus / nodes;

    printf("The number of configured cpus in the system is %d\n", ncpus);

    omp_set_dynamic(0);
    omp_set_num_threads(ncpus);

    #pragma omp parallel for
    for (int tid = 0; tid < ncpus; tid++) {
        int sid = tid / cpus_per_node;
        if (numa_run_on_node(sid) != 0) {
            printf("Error: thread configuration on numa node failed\n");
            exit(-1);
        }
    }

    #pragma omp parallel for
    for (int tid = 0; tid < ncpus; tid++) {
        #pragma omp critical
        {
            printf("thread no.%2d:\n", omp_get_thread_num());

            printf("  %d cpus avaliable for use.\n", numa_num_task_cpus());

            printf("  %d nodes avaliable for use: ", numa_num_task_nodes());
            struct bitmask *allow_cpus = numa_get_run_node_mask();
            for (int i = 0; i < ncpus; i++) {
                if (numa_bitmask_isbitset(allow_cpus, i))
                    printf(" %d", i);
            }
            printf(".\n");
        }
    }
}

int main(int argc, char *argv[])
{
    numa_oblivious_test();

    return 0;
}
