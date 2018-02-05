#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include <numa.h>
#include <omp.h>

int tutorial()
{
    int i, k, w, ncpus;
    struct bitmask *cpus;
    int maxnode = numa_num_configured_nodes();

    if (numa_available() < 0)  {
        printf("no numa\n");
        exit(1);
    }
    cpus = numa_allocate_cpumask();
    ncpus = cpus->size;

    for (i = 0; i < maxnode ; i++) {
        if (numa_node_to_cpus(i, cpus) < 0) {
            printf("node %d failed to convert\n",i); 
        }       
        printf("%d: ", i); 
        w = 0;
        for (k = 0; k < ncpus; k++)
            if (numa_bitmask_isbitset(cpus, k))
                printf(" %s%d", w>0?",":"", k);
        putchar('\n');      
    }

    return 0;
}

void example()
{
    int i;

    if (numa_available() < 0)  {
        printf("Error: no numa node avaliable on this system...\n");
        exit(1);
    }

    printf("The number of possible nodes supported in this system is %d,"\
           " so the number of the highest possible node is %d\n",
            numa_num_possible_nodes(),
            numa_max_possible_node());

    int num_nodes = numa_num_configured_nodes();
    printf("The number of configured nodes in this system is %d,"\
           " and the highest node number is %d\n",
            num_nodes,
            numa_max_node());

    struct bitmask *nodes = numa_get_mems_allowed();
    printf("This is the process %d, the node memory can be accessed are",
            (int)getpid());
    for (i = 0; i < num_nodes; ++i) {
        if (numa_bitmask_isbitset(nodes, i))
            printf(" %d", i);
    }
    printf("\n");

    int ncpus = numa_num_configured_cpus();
}
