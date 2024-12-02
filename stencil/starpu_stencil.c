#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <starpu.h> // Inclusion de StarPU

#define ELEMENT_TYPE float

#define DEFAULT_MESH_WIDTH 2000
#define DEFAULT_MESH_HEIGHT 1000
#define DEFAULT_NB_ITERATIONS 100
#define DEFAULT_NB_REPEAT 10

#define STENCIL_WIDTH 3
#define STENCIL_HEIGHT 3

#define TOP_BOUNDARY_VALUE 10
#define BOTTOM_BOUNDARY_VALUE 5
#define LEFT_BOUNDARY_VALUE -10
#define RIGHT_BOUNDARY_VALUE -5

#define MAX_DISPLAY_COLUMNS 20
#define MAX_DISPLAY_LINES 100

#define EPSILON 1e-3

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 128

static const ELEMENT_TYPE stencil_coefs[STENCIL_HEIGHT * STENCIL_WIDTH] =
{
    0,     0.25, 0   ,
    0.25, -1.00, 0.25,
    0,     0.25, 0
};

enum e_initial_mesh_type
{
    initial_mesh_zero = 1,
    initial_mesh_random = 2
};

struct s_settings
{
    int mesh_width;
    int mesh_height;
    enum e_initial_mesh_type initial_mesh_type;
    int nb_iterations;
    int nb_repeat;
    int enable_output;
    int enable_verbose;
};

#define PRINT_ERROR(MSG)                                                    \
    do                                                                      \
    {                                                                       \
        fprintf(stderr, "%s:%d - %s\n", __FILE__, __LINE__, (MSG));         \
        exit(EXIT_FAILURE);                                                 \
    } while (0)

#define IO_CHECK(OP, RET)                   \
    do                                      \
    {                                       \
        if ((RET) < 0)                      \
        {                                   \
            perror((OP));                   \
            exit(EXIT_FAILURE);             \
        }                                   \
    } while (0)

static void usage(void)
{
    fprintf(stderr, "usage: stencil [OPTIONS...]\n");
    fprintf(stderr, "    --mesh-width  MESH_WIDTH\n");
    fprintf(stderr, "    --mesh-height MESH_HEIGHT\n");
    fprintf(stderr, "    --initial-mesh <zero|random>\n");
    fprintf(stderr, "    --nb-iterations NB_ITERATIONS\n");
    fprintf(stderr, "    --nb-repeat NB_REPEAT\n");
    fprintf(stderr, "    --output\n");
    fprintf(stderr, "    --verbose\n");
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

static void init_settings(struct s_settings **pp_settings)
{
    assert(*pp_settings == NULL);
    struct s_settings *p_settings = calloc(1, sizeof(*p_settings));
    if (p_settings == NULL)
    {
        PRINT_ERROR("memory allocation failed");
    }
    p_settings->mesh_width = DEFAULT_MESH_WIDTH;
    p_settings->mesh_height = DEFAULT_MESH_HEIGHT;
    p_settings->initial_mesh_type = initial_mesh_zero;
    p_settings->nb_iterations = DEFAULT_NB_ITERATIONS;
    p_settings->nb_repeat = DEFAULT_NB_REPEAT;
    p_settings->enable_verbose = 0;
    p_settings->enable_output = 0;
    *pp_settings = p_settings;
}

static void parse_cmd_line(int argc, char *argv[], struct s_settings *p_settings)
{
    int i = 1;
    while (i < argc)
    {
        if (strcmp(argv[i], "--mesh-width") == 0)
        {
            i++;
            if (i >= argc)
            {
                usage();
            }
            int value = atoi(argv[i]);
            if (value < STENCIL_WIDTH)
            {
                fprintf(stderr, "invalid MESH_WIDTH argument\n");
                exit(EXIT_FAILURE);
            }
            p_settings->mesh_width = value;
        }
        else if (strcmp(argv[i], "--mesh-height") == 0)
        {
            i++;
            if (i >= argc)
            {
                usage();
            }
            int value = atoi(argv[i]);
            if (value < STENCIL_HEIGHT)
            {
                fprintf(stderr, "invalid MESH_HEIGHT argument\n");
                exit(EXIT_FAILURE);
            }
            p_settings->mesh_height = value;
        }
        else if (strcmp(argv[i], "--initial-mesh") == 0)
        {
            i++;
            if (i >= argc)
            {
                usage();
            }
            if (strcmp(argv[i], "zero") == 0)
            {
                p_settings->initial_mesh_type = initial_mesh_zero;
            }
            else if (strcmp(argv[i], "random") == 0)
            {
                p_settings->initial_mesh_type = initial_mesh_random;
            }
            else
            {
                fprintf(stderr, "invalid initial mesh type\n");
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "--nb-iterations") == 0)
        {
            i++;
            if (i >= argc)
            {
                usage();
            }
            int value = atoi(argv[i]);
            if (value < 1)
            {
                fprintf(stderr, "invalid NB_ITERATIONS argument\n");
                exit(EXIT_FAILURE);
            }
            p_settings->nb_iterations = value;
        }
        else if (strcmp(argv[i], "--nb-repeat") == 0)
        {
            i++;
            if (i >= argc)
            {
                usage();
            }
            int value = atoi(argv[i]);
            if (value < 1)
            {
                fprintf(stderr, "invalid NB_REPEAT argument\n");
                exit(EXIT_FAILURE);
            }
            p_settings->nb_repeat = value;
        }
        else if (strcmp(argv[i], "--output") == 0)
        {
            p_settings->enable_output = 1;
        }
        else if (strcmp(argv[i], "--verbose") == 0)
        {
            p_settings->enable_verbose = 1;
        }
        else
        {
            usage();
        }

        i++;
    }

    if (p_settings->enable_output)
    {
        p_settings->nb_repeat = 1;
        if (p_settings->nb_iterations > 100)
        {
            p_settings->nb_iterations = 100;
        }
    }
}

static void delete_settings(struct s_settings **pp_settings)
{
    assert(*pp_settings != NULL);
    free(*pp_settings);
    *pp_settings = NULL;
}

static void allocate_mesh(ELEMENT_TYPE **pp_mesh, struct s_settings *p_settings)
{
    assert(*pp_mesh == NULL);
    ELEMENT_TYPE *p_mesh = calloc(p_settings->mesh_width * p_settings->mesh_height, sizeof(*p_mesh));
    if (p_mesh == NULL)
    {
        PRINT_ERROR("memory allocation failed");
    }
    *pp_mesh = p_mesh;
}

static void delete_mesh(ELEMENT_TYPE **pp_mesh)
{
    assert(*pp_mesh != NULL);
    free(*pp_mesh);
    *pp_mesh = NULL;
}

static void init_mesh_zero(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int x;
    int y;
    for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
    {
        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
            p_mesh[y * p_settings->mesh_width + x] = 0;
        }
    }
}

static void init_mesh_random(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int x;
    int y;
    for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
    {
        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
            ELEMENT_TYPE value = ((ELEMENT_TYPE)rand() / (ELEMENT_TYPE)RAND_MAX) * 20 - 10;
            p_mesh[y * p_settings->mesh_width + x] = value;
        }
    }
}

static void init_mesh_values(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    switch (p_settings->initial_mesh_type)
    {
    case initial_mesh_zero:
        init_mesh_zero(p_mesh, p_settings);
        break;

    case initial_mesh_random:
        init_mesh_random(p_mesh, p_settings);
        break;

    default:
        PRINT_ERROR("invalid initial mesh type");
    }
}

static void copy_mesh(ELEMENT_TYPE *p_dst_mesh, const ELEMENT_TYPE *p_src_mesh, struct s_settings *p_settings)
{
    memcpy(p_dst_mesh, p_src_mesh, p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_dst_mesh));
}

static void apply_boundary_conditions(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int x;
    int y;

    // Apply boundary conditions at the top and bottom
    for (x = 0; x < p_settings->mesh_width; x++)
    {
        for (y = 0; y < margin_y; y++)
        {
            p_mesh[y * p_settings->mesh_width + x] = TOP_BOUNDARY_VALUE;
            p_mesh[(p_settings->mesh_height - 1 - y) * p_settings->mesh_width + x] = BOTTOM_BOUNDARY_VALUE;
        }
    }

    // Apply boundary conditions on the left and right
    for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
    {
        for (x = 0; x < margin_x; x++)
        {
            p_mesh[y * p_settings->mesh_width + x] = LEFT_BOUNDARY_VALUE;
            p_mesh[y * p_settings->mesh_width + (p_settings->mesh_width - 1 - x)] = RIGHT_BOUNDARY_VALUE;
        }
    }
}

static void print_settings_csv_header(void)
{
    printf("mesh_width,mesh_height,nb_iterations,nb_repeat");
}

static void print_settings_csv(struct s_settings *p_settings)
{
    printf("%d,%d,%d,%d", p_settings->mesh_width, p_settings->mesh_height, p_settings->nb_iterations, p_settings->nb_repeat);
}

static void print_results_csv_header(void)
{
    printf("rep,timing,check_status");
}

static void print_results_csv(int rep, double timing_in_seconds, int check_status)
{
    printf("%d,%le,%d", rep, timing_in_seconds, check_status);
}

static void print_csv_header(void)
{
    print_settings_csv_header();
    printf(",");
    print_results_csv_header();
    printf("\n");
}

static void print_mesh(const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    int x;
    int y;

    printf("[\n");
    for (y = 0; y < p_settings->mesh_height; y++)
    {
        if (y >= MAX_DISPLAY_LINES)
        {
            printf("...\n");
            break;
        }
        printf("[%03d: ", y);
        for (x = 0; x < p_settings->mesh_width; x++)
        {
            if (x >= MAX_DISPLAY_COLUMNS)
            {
                printf("...");
                break;
            }
            printf(" %+8.2lf", (double)p_mesh[y * p_settings->mesh_width + x]);
        }
        printf("]\n");
    }
    printf("]");
}

static void write_mesh_to_file(FILE *file, const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    int x;
    int y;
    int ret;

    for (y = 0; y < p_settings->mesh_height; y++)
    {
        for (x = 0; x < p_settings->mesh_width; x++)
        {
            if (x > 0)
            {
                ret = fprintf(file, ",");
                IO_CHECK("fprintf", ret);
            }

            ret = fprintf(file, "%lf", (double)p_mesh[y * p_settings->mesh_width + x]);
            IO_CHECK("fprintf", ret);
        }

        ret = fprintf(file, "\n");
        IO_CHECK("fprintf", ret);
    }
}

static void naive_stencil_func(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int x;
    int y;

    ELEMENT_TYPE *p_temporary_mesh = malloc(p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_mesh));
    if (!p_temporary_mesh)
    {
        PRINT_ERROR("memory allocation failed");
    }

    for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
    {
        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
            ELEMENT_TYPE value = p_mesh[y * p_settings->mesh_width + x];
            int stencil_x, stencil_y;
            for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
            {
                for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
                {
                    value +=
                        p_mesh[(y + stencil_y - margin_y) * p_settings->mesh_width + (x + stencil_x - margin_x)] *
                        stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                }
            }
            p_temporary_mesh[y * p_settings->mesh_width + x] = value;
        }
    }

    for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
    {
        memcpy(&p_mesh[y * p_settings->mesh_width + margin_x],
               &p_temporary_mesh[y * p_settings->mesh_width + margin_x],
               (p_settings->mesh_width - 2 * margin_x) * sizeof(ELEMENT_TYPE));
    }

    free(p_temporary_mesh);
}

// Structure for codelet arguments
struct stencil_args {
    int width;
    int height;
    int ld;
};

// Codelet function
static void stencil_cpu_func(void *buffers[], void *cl_arg) {
    // Retrieve pointers to matrices
    ELEMENT_TYPE *p_mesh = (ELEMENT_TYPE*) STARPU_MATRIX_GET_PTR(buffers[0]);
    ELEMENT_TYPE *p_temporary_mesh = (ELEMENT_TYPE*) STARPU_MATRIX_GET_PTR(buffers[1]);

    // Retrieve arguments
    struct stencil_args *args = (struct stencil_args*) cl_arg;

    int ld = args->ld;
    int width = args->width;
    int height = args->height;
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;

    // Loops with local indices
    for(int y = margin_y; y < height + margin_y; y++) {
        for(int x = margin_x; x < width + margin_x; x++) {
            ELEMENT_TYPE value = p_mesh[y * ld + x];
            for(int stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++) {
                for(int stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++) {
                    int neighbor_y = y + stencil_y - margin_y;
                    int neighbor_x = x + stencil_x - margin_x;
                    value += p_mesh[neighbor_y * ld + neighbor_x] 
                             * stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                }
            }
            p_temporary_mesh[y * ld + x] = value;
        }
    }
}

// Define the codelet
static struct starpu_codelet stencil_codelet = {
    .where = STARPU_CPU,
    .cpu_funcs = { stencil_cpu_func, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_W }, // p_mesh en lecture, p_temporary_mesh en écriture
};

// StarPU stencil function
void starpu_stencil_func(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings) {
    // Allocate temporary memory
    ELEMENT_TYPE *p_temporary_mesh = malloc(p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_mesh));
    if (!p_temporary_mesh)
    {
        PRINT_ERROR("memory allocation failed");
    }

    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;

    // Create tasks
    for(int y = margin_y; y < p_settings->mesh_height - margin_y; y += BLOCK_SIZE_Y) {
        for(int x = margin_x; x < p_settings->mesh_width - margin_x; x += BLOCK_SIZE_X) {
            int width = (x + BLOCK_SIZE_X > p_settings->mesh_width - margin_x) ? 
                        (p_settings->mesh_width - margin_x - x) : BLOCK_SIZE_X;
            int height = (y + BLOCK_SIZE_Y > p_settings->mesh_height - margin_y) ? 
                         (p_settings->mesh_height - margin_y - y) : BLOCK_SIZE_Y;

            // Adjusted start positions to include margins
            int x_start_with_margin = x - margin_x;
            int y_start_with_margin = y - margin_y;
            int width_with_margin = width + 2 * margin_x;
            int height_with_margin = height + 2 * margin_y;

            // Create sub-data handles
            starpu_data_handle_t mesh_block_handle, temp_mesh_block_handle;

            starpu_matrix_data_register(&mesh_block_handle, STARPU_MAIN_RAM,
                (uintptr_t)(p_mesh + y_start_with_margin * p_settings->mesh_width + x_start_with_margin),
                p_settings->mesh_width, // leading dimension
                width_with_margin,
                height_with_margin,
                sizeof(ELEMENT_TYPE));

            starpu_matrix_data_register(&temp_mesh_block_handle, STARPU_MAIN_RAM,
                (uintptr_t)(p_temporary_mesh + y_start_with_margin * p_settings->mesh_width + x_start_with_margin),
                p_settings->mesh_width, // leading dimension
                width_with_margin,
                height_with_margin,
                sizeof(ELEMENT_TYPE));

            // Initialize codelet arguments
            struct stencil_args args;
            args.width = width;
            args.height = height;
            args.ld = p_settings->mesh_width;

            // Submit task
            struct starpu_task *task = starpu_task_create();
            task->cl = &stencil_codelet;
            task->handles[0] = mesh_block_handle;
            task->handles[1] = temp_mesh_block_handle;
            task->cl_arg = malloc(sizeof(args));
            memcpy(task->cl_arg, &args, sizeof(args));
            task->cl_arg_size = sizeof(args);
            task->cl_arg_free = 1; // StarPU will free cl_arg

            int ret = starpu_task_submit(task);
            if (ret != 0) {
                fprintf(stderr, "Erreur lors de la soumission de la tâche StarPU\n");
                exit(EXIT_FAILURE);
            }

            // Unregister the sub-data handles
            starpu_data_unregister(mesh_block_handle);
            starpu_data_unregister(temp_mesh_block_handle);
        }
    }

    // Wait for tasks to finish
    starpu_task_wait_for_all();

    // Copy back only the inner elements
    for(int y = margin_y; y < p_settings->mesh_height - margin_y; y++) {
        memcpy(&p_mesh[y * p_settings->mesh_width + margin_x],
               &p_temporary_mesh[y * p_settings->mesh_width + margin_x],
               (p_settings->mesh_width - 2 * margin_x) * sizeof(ELEMENT_TYPE));
    }

    free(p_temporary_mesh);
}

static void run(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    int i;
    for (i = 0; i < p_settings->nb_iterations; i++)
    {
        starpu_stencil_func(p_mesh, p_settings);

        if (p_settings->enable_output)
        {
            char filename[32];
            snprintf(filename, 32, "run_mesh_%03d.csv", i);
            FILE *file = fopen(filename, "w");
            if (file == NULL)
            {
                perror("fopen");
                exit(EXIT_FAILURE);
            }
            write_mesh_to_file(file, p_mesh, p_settings);
            fclose(file);
        }

        if (p_settings->enable_verbose)
        {
            printf("mesh after iteration %d\n", i);
            print_mesh(p_mesh, p_settings);
            printf("\n\n");
        }

        // Reapply boundary conditions to maintain them after each iteration
        apply_boundary_conditions(p_mesh, p_settings);
    }
}

static int check(const ELEMENT_TYPE *p_mesh, ELEMENT_TYPE *p_mesh_copy, struct s_settings *p_settings)
{
    int i;
    for (i = 0; i < p_settings->nb_iterations; i++)
    {
        naive_stencil_func(p_mesh_copy, p_settings);

        if (p_settings->enable_output)
        {
            char filename[32];
            snprintf(filename, 32, "check_mesh_%03d.csv", i);
            FILE *file = fopen(filename, "w");
            if (file == NULL)
            {
                perror("fopen");
                exit(EXIT_FAILURE);
            }
            write_mesh_to_file(file, p_mesh_copy, p_settings);
            fclose(file);
        }

        if (p_settings->enable_verbose)
        {
            printf("check mesh after iteration %d\n", i);
            print_mesh(p_mesh_copy, p_settings);
            printf("\n\n");
        }

        // Reapply boundary conditions to maintain them after each iteration
        apply_boundary_conditions(p_mesh_copy, p_settings);
    }

    int check_status = 0;
    int x;
    int y;
    for (y = 0; y < p_settings->mesh_height; y++)
    {
        for (x = 0; x < p_settings->mesh_width; x++)
        {
            ELEMENT_TYPE diff = fabs(p_mesh[y * p_settings->mesh_width + x] - p_mesh_copy[y * p_settings->mesh_width + x]);
            if (diff > EPSILON)
            {
                fprintf(stderr, "check failed [x: %d, y: %d]: run = %lf, check = %lf\n", x, y,
                        p_mesh[y * p_settings->mesh_width + x],
                        p_mesh_copy[y * p_settings->mesh_width + x]);
                check_status = 1;
            }
        }
    }

    return check_status;
}

int main(int argc, char *argv[])
{
    struct s_settings *p_settings = NULL;

    init_settings(&p_settings);
    parse_cmd_line(argc, argv, p_settings);

    ELEMENT_TYPE *p_mesh = NULL;
    allocate_mesh(&p_mesh, p_settings);

    ELEMENT_TYPE *p_mesh_copy = NULL;
    allocate_mesh(&p_mesh_copy, p_settings);

    // Initialize StarPU once here
    int ret = starpu_init(NULL);
    if (ret != 0) {
        fprintf(stderr, "Erreur d'initialisation de StarPU\n");
        exit(EXIT_FAILURE);
    }

    // Perform computations
    {
        if (!p_settings->enable_verbose)
        {
            print_csv_header();
        }

        int rep;
        for (rep = 0; rep < p_settings->nb_repeat; rep++)
        {
            if (p_settings->enable_verbose)
            {
                printf("repeat %d\n", rep);
            }

            init_mesh_values(p_mesh, p_settings);
            apply_boundary_conditions(p_mesh, p_settings);
            copy_mesh(p_mesh_copy, p_mesh, p_settings);

            if (p_settings->enable_verbose)
            {
                printf("initial mesh\n");
                print_mesh(p_mesh, p_settings);
                printf("\n\n");
            }

            struct timespec timing_start, timing_end;
            clock_gettime(CLOCK_MONOTONIC, &timing_start);
            run(p_mesh, p_settings);
            clock_gettime(CLOCK_MONOTONIC, &timing_end);
            double timing_in_seconds = (timing_end.tv_sec - timing_start.tv_sec) + 1.0e-9 * (timing_end.tv_nsec - timing_start.tv_nsec);

            int check_status = check(p_mesh, p_mesh_copy, p_settings);

            if (p_settings->enable_verbose)
            {
                print_csv_header();
            }
            print_settings_csv(p_settings);
            printf(",");
            print_results_csv(rep, timing_in_seconds, check_status);
            printf("\n");
        }
    }

    // Finalize StarPU once here
    starpu_shutdown();

    delete_mesh(&p_mesh_copy);
    delete_mesh(&p_mesh);
    delete_settings(&p_settings);

    return 0;
}

