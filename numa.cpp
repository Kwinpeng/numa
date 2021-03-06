/****************************************************************
 * FileName: numa.cpp
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 * Todolist:
 *   1. test numa
 *   2. vectorization [avx2]
 *   3. implement two-node numa-aware accumulation
 *   4. loop fusion
 *
 ****************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include <numa.h>
#include <omp.h>

#include "timer.hpp"

#include "emmintrin.h"
#include "immintrin.h"

//////////////////////////////////////////////////////////////////

/* Global configuration */
const int pf  = 2;
const int dim = 280 * pf;
const int batchsize = 29093774;

const int volsize = dim * dim * (dim / 2 + 1);
const size_t total_mem = (batchsize * 3 * 2) * sizeof(double)
                        + batchsize * sizeof(int);

#define ABS(x) ((x) < 0 ? (-(x)) : (x))

//////////////////////////////////////////////////////////////////

/***
 * @breif Search for the numa partition boundary in the download
 *        coordinate stream, using this to distribute task in the
 *        following accumulation.
 * 
 * @param numasize: single node memory size, which means the number
 *                  of elements a numa node holds.
 */
inline int search(int numasize, int *coords, int batchsize)
{
    int left = 0, right = batchsize;
    while (left < right) {
        int mid = (left + right) / 2;
        if (numasize < coords[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return right;
}

void read_data(const int size,
               int *_coords,
               double *_voxels_real,
               double *_voxels_imag,
               double *_voxels_weit)
{
    FILE *fp;

    assert((fp = fopen("data/coor.dat", "rb")) != NULL);
    assert(size == fread(_coords, sizeof(int), size, fp));
    fclose(fp);

    assert((fp = fopen("data/real.dat", "rb")) != NULL);
    assert(size == fread(_voxels_real, sizeof(double), size, fp));
    fclose(fp);

    assert((fp = fopen("data/imag.dat", "rb")) != NULL);
    assert(size == fread(_voxels_imag, sizeof(double), size, fp));
    fclose(fp);

    assert((fp = fopen("data/weit.dat", "rb")) != NULL);
    assert(size == fread(_voxels_weit, sizeof(double), size, fp));
    fclose(fp);
}

int read_data(const int half_volsize,
              const int length,
              int    *_coords[],
              double *_voxels_real[],
              double *_voxels_imag[],
              double *_voxels_weit[])
{
    FILE *fp;
    int *coords = (int*)malloc(length * sizeof(int));

    assert((fp = fopen("data/coor.dat", "rb")) != NULL);
    assert(length == fread(coords, sizeof(int), length, fp));
    fclose(fp);

    const int partition_boundary = search(half_volsize, coords, length);

    /* partition numa coords */
    memcpy(_coords[0], coords, partition_boundary * sizeof(int));
    memcpy(_coords[1], coords + partition_boundary,
           (length - partition_boundary) * sizeof(int));

    for (int i = 0; i < length; i++) {
        if (i < partition_boundary) {
            if (_coords[0][i] == 0)
                printf("error: node 0 index %d, val %d\n",
                        i, _coords[0][i]);
        } else {
            if (_coords[1][i - partition_boundary] == 0)
                printf("error: node 1 index %d, val %d\n",
                        i, _coords[1][i - partition_boundary]);
        }
    }

    /* partition real imag and weit*/
    assert((fp = fopen("data/real.dat", "rb")) != NULL);
    assert(partition_boundary == fread(_voxels_real[0],
                sizeof(double), partition_boundary, fp));
    assert((length - partition_boundary) == fread(_voxels_real[1],
                sizeof(double), (length - partition_boundary), fp));
    fclose(fp);

    assert((fp = fopen("data/imag.dat", "rb")) != NULL);
    assert(partition_boundary == fread(_voxels_imag[0],
                sizeof(double), partition_boundary, fp));
    assert((length - partition_boundary) == fread(_voxels_imag[1],
                sizeof(double), (length - partition_boundary), fp));

    assert((fp = fopen("data/weit.dat", "rb")) != NULL);
    assert(partition_boundary == fread(_voxels_weit[0],
                sizeof(double), partition_boundary, fp));
    assert((length - partition_boundary) == fread(_voxels_weit[1],
                sizeof(double), (length - partition_boundary), fp));
    fclose(fp);

    free(coords);

    return partition_boundary;
}

void coord_voxel_analytics()
{
    printf("========================================\n");
    printf("Download coord-voxel stream analytics\n");
    printf("----------------------------------------\n");

    /* allocate memory */
    int *_coords = (int*)malloc(batchsize * sizeof(int));
    double *_voxels_real = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_imag = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_weit = (double*)malloc(batchsize * sizeof(double));

    /* read data from file */
    read_data(batchsize, _coords, _voxels_real, _voxels_imag, _voxels_weit);

    Timer timer;
    timer.start();

    /* binary search for numa partition boundary */
    const int numa_boundary = search(volsize / 2, _coords, batchsize);

    timer.interval_timing("search for numa binary");

    printf(" download batchsize is %d\n", batchsize);
    printf(" searched index in coords: %d\n", numa_boundary);
    for (int i = numa_boundary - 5; i < numa_boundary + 5; i++) {
        printf("   coords stream[%d]: %d\n", i, _coords[i]);
    }

    printf(" half volume index is %d\n", volsize / 2);
}

bool accumulate_check(double *volume, double *weight)
{
    double *_volume = (double*)calloc(volsize * 2, sizeof(double));
    double *_weight = (double*)calloc(volsize * 2, sizeof(double));

    int *_coords         = (int*)malloc(batchsize * sizeof(int));
    double *_voxels_real = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_imag = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_weit = (double*)malloc(batchsize * sizeof(double));

    read_data(batchsize, _coords, _voxels_real, _voxels_imag, _voxels_weit);

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

    /* check value */
    bool ret = true;

    for (int i = 0; i < volsize; i++) {
        if (ABS(volume[i] - _volume[i]) > 1e-8) {
            printf("volume error: value %16.10f expext %16.10f\n",
                    volume[i], _volume[i]);
            ret = false;
            break;
        }
    }

    for (int i = 0; i < volsize; i++) {
        if (ABS(weight[i] - _weight[i]) > 1e-8) {
            printf("weight error: value %16.9f expext %16.9f\n",
                    weight[i], _weight[i]);
            ret = false;
            break;
        }
    }

    free(_coords);
    free(_voxels_real);
    free(_voxels_imag);
    free(_voxels_weit);
    free(_volume);
    free(_weight);

    return ret;
}

bool numa_accumulate_check(double *volume[], double *weight[],
                           int half_volsize, int nodes)
{
    double *_volume = (double*)calloc(volsize * 2, sizeof(double));
    double *_weight = (double*)calloc(volsize * 2, sizeof(double));

    /* check not NULL pointer */
    for (int nid = 0; nid < nodes; nid++) {
        assert(volume[nid] != NULL);
        assert(weight[nid] != NULL);

        memcpy(_volume + nid * half_volsize,
               volume[nid],
               half_volsize * 2 * sizeof(double));
        memcpy(_weight + nid * half_volsize,
               weight[nid],
               half_volsize * 2 * sizeof(double));
    }

    bool ret = accumulate_check(_volume, _weight);

    free(_volume);
    free(_weight);

    return ret;
}

void system_detect(int& ncpus, int& nodes, int& cpus_per_node)
{
    if (numa_available() < 0)  {
        printf("Error: no numa node avaliable on this system...\n");
        exit(1);
    }

    /* detect cpus and numa nodes */
    ncpus = numa_num_configured_cpus();
    nodes = numa_num_configured_nodes();
    cpus_per_node = ncpus / nodes;

    printf(" System configuration:\n");
    printf("   # of numa node: %d\n", nodes);
    printf("   # of cpus: %d\n", ncpus);
    #pragma omp parallel for
    for (int i = 0; i < ncpus; i++) {
        if (omp_get_thread_num() == 0 && i == 0)
            printf("   # of omp threads: %d\n", omp_get_num_threads());
    }
}

void system_detect()
{
    int ncpus, nodes, cpus_per_node;
    system_detect(ncpus, nodes, cpus_per_node);
}

///////////////////////////////////////////////////////////////////////////

void numa_oblivious_test()
{
    printf("========================================\n");
    printf("NUMA oblivious test\n");
    printf("----------------------------------------\n");

    system_detect();

    double *_volume = (double*)calloc(volsize * 2, sizeof(double));
    double *_weight = (double*)calloc(volsize * 2, sizeof(double));

    int *_coords = (int*)malloc(batchsize * sizeof(int));
    double *_voxels_real = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_imag = (double*)malloc(batchsize * sizeof(double));
    double *_voxels_weit = (double*)malloc(batchsize * sizeof(double));

    Timer timer;
    timer.start();

    read_data(batchsize, _coords, _voxels_real, _voxels_imag, _voxels_weit);

    timer.interval_timing("Data reading");

#if 0
    /* baseline */
    int pos = 0;
    #pragma omp parallel for
    for (int n = 0; n < batchsize; ++n) {
        int current = pos + n;

        int index = _coords[current];

        //#pragma omp atomic
        //_volume[index * 2] += _voxels_real[current];
        //#pragma omp atomic
        //_volume[index * 2 + 1] += _voxels_imag[current];
        //#pragma omp atomic
        //_weight[index * 2] += _voxels_weit[current];

        /* bandwidth test */ 
        _volume[index * 2] = _voxels_real[current];
        _volume[index * 2 + 1] = _voxels_imag[current];
        _weight[index * 2] = _voxels_weit[current];
    }
#else
#define VEC_TILE 2
    /* vectorization */ 
    __m256i reg_mask = _mm256_set_epi64x(0, -1, 0, -1);

    #pragma omp parallel for
    for (int n = 0; n < batchsize / VEC_TILE; n += 2) {
        int index = _coords[n];

        /* ------------------------------------------------------------- *
         * Register layout:
         *
         * [255:192]|[191:128]|[127:64]|[63:0] = [im2]|[im1]|[rl2]|[rl1]
         * ------------------------------------------------------------- */

        /* load */
        __m256d reg_im_rl = _mm256_loadu2_m128d(_voxels_imag + n,
                                                _voxels_real + n);

        __m256d reg_voxel = _mm256_loadu_pd(_volume + index * 2);

        /* add */
        reg_voxel = _mm256_add_pd(reg_voxel, reg_im_rl);

        /* store */
        _mm256_storeu_pd(_volume + index * 2, reg_voxel);

        //_weight[index * 2] += _voxels_weit[n];

        /* load 2 weights with imaginary part zero */
        __m256d reg_vol_weit = _mm256_maskload_pd(_weight + index * 2, reg_mask);

        /* load weights from _voxel steram */
        __m128d reg_vxl_weit_0 = _mm_load_sd(_voxels_weit + n);
        __m128d reg_vxl_weit_1 = _mm_load_sd(_voxels_weit + n + 1);
        __m256d reg_vxl_weit = _mm256_set_m128d(reg_vxl_weit_1, reg_vxl_weit_0);

        /* add */
        reg_vol_weit = _mm256_add_pd(reg_vol_weit, reg_vxl_weit);

        /* store */
        _mm256_storeu_pd(_voxels_weit + n, reg_vol_weit);
    }
#endif

    double time_elapse = timer.interval_timing("Accumulating");

    printf("Memory throughput: %.4f GB/s\n", (total_mem >> 30) / (time_elapse / 1e6));

    free(_volume);
    free(_weight);
    free(_coords);
    free(_voxels_real);
    free(_voxels_imag);
    free(_voxels_weit);
}

void numa_node_local_test()
{
    printf("========================================\n");
    printf("NUMA node local test\n");
    printf("----------------------------------------\n");

    int ncpus, nodes, cpus_per_node;
    system_detect(ncpus, nodes, cpus_per_node);

    /* specify the node */
    const int node = 0;

    /* detect cpus on this node */
    int num_cpus_onnode = 0;
    struct bitmask *cpus_onnode = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpus_onnode);
    for (int cpuid = 0; cpuid < ncpus; ++cpuid) {
        if (numa_bitmask_isbitset(cpus_onnode , cpuid))
            num_cpus_onnode++;
    }

    printf("%d threads in use\n", num_cpus_onnode);

    /* configure omp threads */
    omp_set_dynamic(0);
    omp_set_num_threads(ncpus);

    #pragma omp parallel for
    for (int tid = 0; tid < ncpus; tid++) {
        int nid = tid / num_cpus_onnode;
        if (numa_run_on_node(nid) != 0) {
            printf("Error: thread configuration on specified node failed\n");
            exit(-1);
        }
    }

    /* allocate numa-aware memory on node local */
    double *numa_volume, *numa_weight;
    numa_volume = (double*)numa_alloc_onnode(volsize * 2 * sizeof(double), node);
    numa_weight = (double*)numa_alloc_onnode(volsize * 2 * sizeof(double), node);

    /* clear */
    for (int i = 0; i < volsize; i++) {
        numa_volume[i * 2]     = 0.0;
        numa_volume[i * 2 + 1] = 0.0;
        numa_weight[i * 2]     = 0.0;
        numa_weight[i * 2 + 1] = 0.0;
    }

    int *numa_coords = (int*)numa_alloc_onnode(batchsize * sizeof(int), node);
    
    double *numa_vxls_real, *numa_vxls_imag, *numa_vxls_weit;
    numa_vxls_real = (double*)numa_alloc_onnode(batchsize * sizeof(double), node);
    numa_vxls_imag = (double*)numa_alloc_onnode(batchsize * sizeof(double), node);
    numa_vxls_weit = (double*)numa_alloc_onnode(batchsize * sizeof(double), node);

    Timer timer;
    timer.start();

    /* read form file */
    read_data(batchsize, numa_coords, numa_vxls_real, numa_vxls_imag, numa_vxls_weit);

    timer.interval_timing("Data reading");

    /* perform accumulation */
    int pos = 0;
    #pragma omp parallel for
    for (int n = 0; n < batchsize; ++n) {
        int current = pos + n;

        int index = numa_coords[current];

        #pragma omp atomic
        numa_volume[index * 2] += numa_vxls_real[current];
        #pragma omp atomic
        numa_volume[index * 2 + 1] += numa_vxls_imag[current];
        #pragma omp atomic
        numa_weight[index * 2] += numa_vxls_weit[current];
    }

    double time_elapse = timer.interval_timing("Accumulating");

    printf("Memory throughput: %.4f GB/s\n", (total_mem >> 30) / (time_elapse / 1e6));

    /* check */
    accumulate_check(numa_volume, numa_weight);

    numa_free(numa_volume, volsize * 2 * sizeof(double));
    numa_free(numa_weight, volsize * 2 * sizeof(double));
    numa_free(numa_vxls_real, batchsize * sizeof(double));
    numa_free(numa_vxls_imag, batchsize * sizeof(double));
    numa_free(numa_vxls_weit, batchsize * sizeof(double));
}

void numa_aware_two_node()
{
    printf("========================================\n");
    printf("NUMA aware two node test\n");
    printf("----------------------------------------\n");

#define NUM_NODES 2

    int ncpus, nodes, cpus_per_node;
    system_detect(ncpus, nodes, cpus_per_node);

    /* configure omp threads */
    omp_set_dynamic(0);
    omp_set_num_threads(ncpus);

    #pragma omp parallel for
    for (int tid = 0; tid < ncpus; tid++) {
        int nid = tid / cpus_per_node;
        if (numa_run_on_node(nid) != 0) {
            printf("Error: thread configuration on specified node failed\n");
            exit(-1);
        }
    }

    /* allocate numa-aware memory on node local */
    const int partition_volsize = volsize / 2;

    double *numa_volume[NUM_NODES], *numa_weight[NUM_NODES];
    int    *numa_coords[NUM_NODES];
    double *numa_vxls_real[NUM_NODES];
    double *numa_vxls_imag[NUM_NODES];
    double *numa_vxls_weit[NUM_NODES];

    for (int nid = 0; nid < NUM_NODES; nid++) {
        numa_volume[nid] = (double*)numa_alloc_onnode(
                                    partition_volsize * 2 * sizeof(double), nid);
        numa_weight[nid] = (double*)numa_alloc_onnode(
                                    partition_volsize * 2 * sizeof(double), nid);

        /* clear */
        for (int i = 0; i < partition_volsize; i++) {
            numa_volume[nid][i * 2]     = 0.0;
            numa_volume[nid][i * 2 + 1] = 0.0;
            numa_weight[nid][i * 2]     = 0.0;
            numa_weight[nid][i * 2 + 1] = 0.0;
        }

        numa_coords[nid]    = (int*)numa_alloc_onnode(
                                       batchsize * sizeof(int), nid);
        numa_vxls_real[nid] = (double*)numa_alloc_onnode(
                                       batchsize * sizeof(double), nid);
        numa_vxls_imag[nid] = (double*)numa_alloc_onnode(
                                       batchsize * sizeof(double), nid);
        numa_vxls_weit[nid] = (double*)numa_alloc_onnode(
                                       batchsize * sizeof(double), nid);
    }

    /* allocate numa-aware task distribution index array */
    int *task[NUM_NODES];
    for (int nid = 0; nid < NUM_NODES; ++nid) {
        task[nid] = (int*)numa_alloc_onnode(cpus_per_node * 2 * sizeof(int), nid);
    }

    Timer timer;
    timer.start();

    /* read data from file */
    const int partition_boundary = read_data(partition_volsize, batchsize, 
              numa_coords, numa_vxls_real, numa_vxls_imag, numa_vxls_weit);

    printf(" partition boundary: %d\n", partition_boundary);

    /* global shared partition info [numa oblivious] */
    int partition[3];
    partition[0] = 0;
    partition[1] = partition_boundary;
    partition[2] = batchsize;

    timer.interval_timing(" Data reading");

#if 0
    /* perform accumulation */
    #pragma omp parallel
    {
        int nid = omp_get_thread_num() / cpus_per_node;

        #pragma omp for nowait
        for (int n = 0; n < partition_boundary; ++n) {

            int index = numa_coords[0][n];

            numa_volume[0][index * 2]     += numa_vxls_real[0][n];
            numa_volume[0][index * 2 + 1] += numa_vxls_imag[0][n];
            numa_weight[0][index * 2]     += numa_vxls_weit[0][n];
        }

        #pragma omp for
        for (int n = 0; n < batchsize - partition_boundary; ++n) {

            int index = numa_coords[1][n];

            numa_volume[1][index * 2]     += numa_vxls_real[1][n];
            numa_volume[1][index * 2 + 1] += numa_vxls_imag[1][n];
            numa_weight[1][index * 2]     += numa_vxls_weit[1][n];
        }
    }
#else
    /*************************************************************
     * Task distribution between two groups of NUMA threads.
     *
     * The i-th thread accumulate the voxels with the coordinates
     * read from coords[task[i]] to coords[task[i+1]].
     *
     * Allocate the task distribution index array in numa-aware
     * style.
     *************************************************************/

    /* allocate NUMA aware task */
    for (int nid = 0; nid < nodes; nid++) {
        int tile = (partition[nid + 1] - partition[nid]) / cpus_per_node;
        for (int tid = 0; tid < cpus_per_node; tid++) {
            /* this tid is node-local index */
            task[nid][tid] = tile * tid;
        }
        /* mark the end of the last thread */
        task[nid][cpus_per_node] = partition[nid + 1] - partition[nid];
    }

    // FIXME debug info
    for (int nid = 0; nid < nodes; nid++) {
        for (int tid = 0; tid < cpus_per_node; tid++) {
            printf("node %d thread %2d task [%12d,%12d), length %10d\n",
                    nid, tid, task[nid][tid], task[nid][tid + 1],
                    task[nid][tid + 1] - task[nid][tid]);
        }
    }

    //#pragma omp parallel
    //{
        /* distribute task */
        //const int tid = omp_get_thread_num() % cpus_per_node;
        //const int nid = omp_get_thread_num() / cpus_per_node;
for (int i = 0; i < 24; i++) {
        const int tid = i % cpus_per_node;
        const int nid = i / cpus_per_node;

        /* do task */
        for (int n = task[nid][tid]; n < task[nid][tid + 1]; ++n) {
            int index = numa_coords[nid][n] - nid * partition_volsize;

            if (index < 0) {
                printf("error: negative index, nid:%d tid:%2d n:%d index:%d\n",
                        nid, tid, n, index);
                continue;
            }

            numa_volume[nid][index * 2]     += numa_vxls_real[nid][n];
            numa_volume[nid][index * 2 + 1] += numa_vxls_imag[nid][n];
            numa_weight[nid][index * 2]     += numa_vxls_weit[nid][n];
        }
}
    //}
#endif

    /* report throughput */
    double time_elapse = timer.interval_timing(" Accumulating");

    printf("Memory throughput: %.4f GB/s\n", (total_mem >> 30) / (time_elapse / 1e6));

    /* check result */
    numa_accumulate_check(numa_volume, numa_weight, partition_volsize, nodes);

    /* numa free */
    for (int nid = 0; nid < NUM_NODES; nid++) {
        numa_free(numa_volume[nid], partition_volsize * 2 * sizeof(double));
        numa_free(numa_weight[nid], partition_volsize * 2 * sizeof(double));

        numa_free(numa_coords[nid],    batchsize * sizeof(int));
        numa_free(numa_vxls_real[nid], batchsize * sizeof(double));
        numa_free(numa_vxls_imag[nid], batchsize * sizeof(double));
        numa_free(numa_vxls_weit[nid], batchsize * sizeof(double));
    }
}

void numa_aware_test()
{
    int ncpus, nodes, cpus_per_node;
    system_detect(ncpus, nodes, cpus_per_node);

    omp_set_dynamic(0);
    omp_set_num_threads(ncpus);

    #pragma omp parallel for
    for (int tid = 0; tid < ncpus; tid++) {
        int nid = tid / cpus_per_node;
        if (numa_run_on_node(nid) != 0) {
            printf("Error: thread configuration on numa node failed\n");
            exit(-1);
        }
    }
}

//////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    //coord_voxel_analytics();

    //numa_oblivious_test();

    numa_node_local_test();

    numa_aware_two_node();

    return 0;
}
