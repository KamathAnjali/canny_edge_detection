/*
 * Sequential single-threaded C reference implementation of Canny edge detection.
 *
 * Based on:
 * M. Horvath Jr., M. Bowers, and S. Alawneh,
 * "Canny Edge Detection on GPU using CUDA,"
 * IEEE 13th Annual Computing and Communication Workshop and Conference (CCWC), 2023.
 *
 * Logical stages:
 *   1. RGB -> grayscale
 *   2. 5x5 Gaussian blur
 *   3. Sobel X/Y -> magnitude + direction
 *   4. Non-maximum suppression
 *   5. Double threshold + hysteresis
 *
 * I/O:
 *   Input  : binary PPM (P6)
 *   Output : binary PGM (P5)
 *
 * Build:
 *   clang -std=c11 -O0 -Wall -Wextra -o canny_baseline canny_baseline.c -lm
 *   clang -std=c11 -O2 -Wall -Wextra -o canny_baseline_O2 canny_baseline.c -lm
 *
 * Usage:
 *   ./canny_baseline input.ppm output.pgm [low_threshold] [high_threshold]
 *
 * Example:
 *   ./canny_baseline input.ppm output.pgm 50 100
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int width;
    int height;
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
} RGBImage;

typedef struct {
    int width;
    int height;
    double *data;
} GrayImage;

/* ---------------------------- Utilities ---------------------------- */

static void die(const char *message)
{
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) die("memory allocation failed");
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if (!p) die("memory allocation failed");
    return p;
}

static double now_ms(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        die("clock_gettime failed");

    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static void free_rgb(RGBImage *img)
{
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL;
}

static void free_gray(GrayImage *img)
{
    free(img->data);
    img->data = NULL;
}

static GrayImage alloc_gray(int width, int height)
{
    GrayImage img;
    img.width = width;
    img.height = height;
    img.data = xcalloc((size_t)width * (size_t)height, sizeof(double));
    return img;
}

/*
 * Read one whitespace-delimited PPM token while skipping comments.
 * This makes the P6 reader robust to comments anywhere in the header.
 */
static int read_token(FILE *f, char *buf, size_t buf_size)
{
    int c;
    size_t n = 0;

    do {
        c = fgetc(f);
        if (c == '#') {
            do {
                c = fgetc(f);
            } while (c != '\n' && c != EOF);
        }
    } while (c != EOF && (c == ' ' || c == '\t' ||
                          c == '\n' || c == '\r' || c == '\f'));

    if (c == EOF)
        return 0;

    do {
        if (n + 1 < buf_size)
            buf[n++] = (char)c;
        c = fgetc(f);
    } while (c != EOF && c != ' ' && c != '\t' &&
             c != '\n' && c != '\r' && c != '\f' && c != '#');

    buf[n] = '\0';

    if (c == '#') {
        do {
            c = fgetc(f);
        } while (c != '\n' && c != EOF);
    }

    return 1;
}

/* ---------------------------- PPM/PGM I/O ---------------------------- */

static RGBImage read_ppm(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Could not open input file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    char token[64];

    if (!read_token(f, token, sizeof(token)) || strcmp(token, "P6") != 0)
        die("only binary PPM (P6) input is supported");

    if (!read_token(f, token, sizeof(token)))
        die("invalid PPM width");
    int width = atoi(token);

    if (!read_token(f, token, sizeof(token)))
        die("invalid PPM height");
    int height = atoi(token);

    if (!read_token(f, token, sizeof(token)))
        die("invalid PPM max value");
    int maxval = atoi(token);

    if (width <= 0 || height <= 0)
        die("invalid image dimensions");

    if (maxval != 255)
        die("only 8-bit PPM files (max value 255) are supported");

    size_t pixels = (size_t)width * (size_t)height;

    RGBImage img;
    img.width = width;
    img.height = height;
    img.r = xmalloc(pixels);
    img.g = xmalloc(pixels);
    img.b = xmalloc(pixels);

    unsigned char *buffer = xmalloc(3 * pixels);

    size_t bytes_read = fread(buffer, 1, 3 * pixels, f);
    fclose(f);

    if (bytes_read != 3 * pixels) {
        free(buffer);
        free_rgb(&img);
        die("PPM pixel data is incomplete");
    }

    for (size_t i = 0; i < pixels; ++i) {
        img.r[i] = buffer[3 * i];
        img.g[i] = buffer[3 * i + 1];
        img.b[i] = buffer[3 * i + 2];
    }

    free(buffer);
    return img;
}

static void write_pgm(const char *path, const GrayImage *img)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Could not open output file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P5\n%d %d\n255\n", img->width, img->height);

    size_t pixels = (size_t)img->width * (size_t)img->height;
    unsigned char *buffer = xmalloc(pixels);

    for (size_t i = 0; i < pixels; ++i) {
        double value = img->data[i];

        if (value < 0.0) value = 0.0;
        if (value > 255.0) value = 255.0;

        buffer[i] = (unsigned char)lround(value);
    }

    if (fwrite(buffer, 1, pixels, f) != pixels) {
        free(buffer);
        fclose(f);
        die("failed while writing PGM output");
    }

    free(buffer);
    fclose(f);
}

/* ---------------------------- Stage 1 ---------------------------- */

/*
 * RGB luminosity conversion.
 *
 * gray = 0.299 R + 0.587 G + 0.114 B
 */
static GrayImage rgb_to_grayscale(const RGBImage *img)
{
    GrayImage out = alloc_gray(img->width, img->height);

    size_t pixels = (size_t)img->width * (size_t)img->height;

    for (size_t i = 0; i < pixels; ++i) {
        out.data[i] =
            0.299 * (double)img->r[i] +
            0.587 * (double)img->g[i] +
            0.114 * (double)img->b[i];
    }

    return out;
}

/* ---------------------------- Convolution ---------------------------- */

/*
 * Generic naive convolution with zero padding.
 *
 * The reference paper describes its GPU convolution operations as
 * flipped convolutions. Therefore the kernel is accessed in reverse
 * order here as well.
 */
static GrayImage convolve(
    const GrayImage *img,
    const double *kernel,
    int kernel_size)
{
    int half = kernel_size / 2;

    GrayImage out = alloc_gray(img->width, img->height);

    for (int y = 0; y < img->height; ++y) {
        for (int x = 0; x < img->width; ++x) {

            double accumulator = 0.0;

            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {

                    int iy = y + ky - half;
                    int ix = x + kx - half;

                    double pixel = 0.0;

                    /* Zero-padding outside the image. */
                    if (iy >= 0 && iy < img->height &&
                        ix >= 0 && ix < img->width) {

                        pixel = img->data[
                            (size_t)iy * (size_t)img->width + (size_t)ix
                        ];
                    }

                    int flipped_ky = kernel_size - 1 - ky;
                    int flipped_kx = kernel_size - 1 - kx;

                    accumulator +=
                        pixel *
                        kernel[flipped_ky * kernel_size + flipped_kx];
                }
            }

            out.data[
                (size_t)y * (size_t)img->width + (size_t)x
            ] = accumulator;
        }
    }

    return out;
}

/* ---------------------------- Stage 2 ---------------------------- */

/*
 * Gaussian kernel shown in Equation I of the paper.
 *
 * The coefficients sum to 150, hence the normalization by 1/150.
 */
static const double GAUSSIAN_KERNEL[25] = {
     2,  4,  5,  4,  2,
     4,  9, 12,  9,  4,
     5, 12, 15, 12,  5,
     4,  9, 12,  9,  4,
     2,  4,  5,  4,  2
};

static GrayImage gaussian_blur(const GrayImage *img)
{
    double normalized[25];

    for (int i = 0; i < 25; ++i)
        normalized[i] = GAUSSIAN_KERNEL[i] / 150.0;

    return convolve(img, normalized, 5);
}

/* ---------------------------- Stage 3 ---------------------------- */

static const double SOBEL_X[9] = {
     1,  0, -1,
     2,  0, -2,
     1,  0, -1
};

static const double SOBEL_Y[9] = {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
};

static void sobel(
    const GrayImage *img,
    GrayImage *magnitude,
    GrayImage *direction)
{
    GrayImage gx = convolve(img, SOBEL_X, 3);
    GrayImage gy = convolve(img, SOBEL_Y, 3);

    *magnitude = alloc_gray(img->width, img->height);
    *direction = alloc_gray(img->width, img->height);

    size_t pixels = (size_t)img->width * (size_t)img->height;

    for (size_t i = 0; i < pixels; ++i) {
        double x = gx.data[i];
        double y = gy.data[i];

        magnitude->data[i] = hypot(x, y);
        direction->data[i] = atan2(y, x);
    }

    free_gray(&gx);
    free_gray(&gy);
}

/* ---------------------------- Stage 4 ---------------------------- */

/*
 * Quantize the gradient direction into four orientations:
 *
 *   0 degrees
 *   45 degrees
 *   90 degrees
 *   135 degrees
 *
 * This is a standard Canny implementation choice. The paper describes
 * selecting the two neighboring pixels along the gradient direction,
 * but does not specify the complete numerical quantization rule.
 */
static GrayImage non_maximum_suppression(
    const GrayImage *magnitude,
    const GrayImage *direction)
{
    int w = magnitude->width;
    int h = magnitude->height;

    GrayImage out = alloc_gray(w, h);

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {

            int idx = y * w + x;

            double angle =
                direction->data[idx] * 180.0 / M_PI;

            if (angle < 0.0)
                angle += 180.0;

            double current = magnitude->data[idx];
            double neighbor1;
            double neighbor2;

            if ((angle >= 0.0 && angle < 22.5) ||
                (angle >= 157.5 && angle < 180.0)) {

                /* 0 degrees: left/right */
                neighbor1 = magnitude->data[idx - 1];
                neighbor2 = magnitude->data[idx + 1];

            } else if (angle < 67.5) {

                /* 45 degrees: NE/SW */
                neighbor1 =
                    magnitude->data[(y - 1) * w + (x + 1)];

                neighbor2 =
                    magnitude->data[(y + 1) * w + (x - 1)];

            } else if (angle < 112.5) {

                /* 90 degrees: up/down */
                neighbor1 =
                    magnitude->data[(y - 1) * w + x];

                neighbor2 =
                    magnitude->data[(y + 1) * w + x];

            } else {

                /* 135 degrees: NW/SE */
                neighbor1 =
                    magnitude->data[(y - 1) * w + (x - 1)];

                neighbor2 =
                    magnitude->data[(y + 1) * w + (x + 1)];
            }

            if (current >= neighbor1 && current >= neighbor2)
                out.data[idx] = current;
            else
                out.data[idx] = 0.0;
        }
    }

    /*
     * The reference CUDA implementation uses a two-pixel border
     * exclusion in its suppression stage. We explicitly zero that
     * border in the sequential reference as well.
     */
    for (int x = 0; x < w; ++x) {
        out.data[x] = 0.0;
        out.data[(h - 1) * w + x] = 0.0;

        if (h > 3) {
            out.data[w + x] = 0.0;
            out.data[(h - 2) * w + x] = 0.0;
        }
    }

    for (int y = 0; y < h; ++y) {
        out.data[y * w] = 0.0;
        out.data[y * w + (w - 1)] = 0.0;

        if (w > 3) {
            out.data[y * w + 1] = 0.0;
            out.data[y * w + (w - 2)] = 0.0;
        }
    }

    return out;
}

/* ---------------------------- Stage 5 ---------------------------- */

#define STRONG 255.0
#define WEAK   127.0

/*
 * Double threshold followed by 8-connected hysteresis.
 *
 * Hysteresis is implemented as a stack-based flood fill starting from
 * strong pixels. Weak pixels connected to a strong component become
 * strong; remaining weak pixels become non-edge.
 */
static GrayImage double_threshold_and_hysteresis(
    const GrayImage *suppressed,
    double low,
    double high)
{
    int w = suppressed->width;
    int h = suppressed->height;
    size_t pixels = (size_t)w * (size_t)h;

    GrayImage out = alloc_gray(w, h);

    /* Double threshold. */
    for (size_t i = 0; i < pixels; ++i) {
        double value = suppressed->data[i];

        if (value >= high)
            out.data[i] = STRONG;
        else if (value >= low)
            out.data[i] = WEAK;
        else
            out.data[i] = 0.0;
    }

    /*
     * Stack-based propagation from all strong pixels.
     * Each pixel is pushed at most once.
     */
    unsigned char *visited = xcalloc(pixels, sizeof(unsigned char));
    size_t *stack = xmalloc(pixels * sizeof(size_t));
    size_t stack_size = 0;

    for (size_t i = 0; i < pixels; ++i) {
        if (out.data[i] == STRONG) {
            visited[i] = 1;
            stack[stack_size++] = i;
        }
    }

    while (stack_size > 0) {
        size_t idx = stack[--stack_size];

        int y = (int)(idx / (size_t)w);
        int x = (int)(idx % (size_t)w);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

                if (dx == 0 && dy == 0)
                    continue;

                int ny = y + dy;
                int nx = x + dx;

                if (ny < 0 || ny >= h || nx < 0 || nx >= w)
                    continue;

                size_t nidx =
                    (size_t)ny * (size_t)w + (size_t)nx;

                if (!visited[nidx] && out.data[nidx] == WEAK) {
                    out.data[nidx] = STRONG;
                    visited[nidx] = 1;
                    stack[stack_size++] = nidx;
                }
            }
        }
    }

    /* Any weak pixel not connected to a strong pixel is discarded. */
    for (size_t i = 0; i < pixels; ++i) {
        if (out.data[i] == WEAK)
            out.data[i] = 0.0;
    }

    free(visited);
    free(stack);

    return out;
}

/* ---------------------------- Main pipeline ---------------------------- */

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "Usage: %s input.ppm output.pgm [low] [high]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    double low = (argc >= 4) ? atof(argv[3]) : 50.0;
    double high = (argc >= 5) ? atof(argv[4]) : 100.0;

    if (low < 0.0 || high < 0.0 || low > high)
        die("thresholds must satisfy 0 <= low <= high");

    /*
     * I/O is deliberately outside the algorithm timing.
     * This makes the reported time represent the Canny computation,
     * rather than file loading/writing.
     */
    RGBImage rgb = read_ppm(input_path);

    double t0, t1;
    double t_gray, t_blur, t_sobel;
    double t_nms, t_threshold;

    /* Stage 1 */
    t0 = now_ms();
    GrayImage gray = rgb_to_grayscale(&rgb);
    t1 = now_ms();
    t_gray = t1 - t0;

    /* Stage 2 */
    t0 = now_ms();
    GrayImage blurred = gaussian_blur(&gray);
    t1 = now_ms();
    t_blur = t1 - t0;

    /* Stage 3 */
    t0 = now_ms();
    GrayImage magnitude;
    GrayImage direction;
    sobel(&blurred, &magnitude, &direction);
    t1 = now_ms();
    t_sobel = t1 - t0;

    /* Stage 4 */
    t0 = now_ms();
    GrayImage suppressed =
        non_maximum_suppression(&magnitude, &direction);
    t1 = now_ms();
    t_nms = t1 - t0;

    /* Stage 5 */
    t0 = now_ms();
    GrayImage edges =
        double_threshold_and_hysteresis(&suppressed, low, high);
    t1 = now_ms();
    t_threshold = t1 - t0;

    double total =
        t_gray + t_blur + t_sobel + t_nms + t_threshold;

    write_pgm(output_path, &edges);

    printf("Resolution: %dx%d\n", rgb.width, rgb.height);
    printf("Thresholds: low=%.2f, high=%.2f\n", low, high);
    printf("\n");
    printf("%-28s %12s\n", "Stage", "Time (ms)");
    printf("------------------------------------------\n");
    printf("%-28s %12.3f\n", "Grayscale", t_gray);
    printf("%-28s %12.3f\n", "Gaussian Blur", t_blur);
    printf("%-28s %12.3f\n", "Sobel Filter", t_sobel);
    printf("%-28s %12.3f\n", "NMS", t_nms);
    printf("%-28s %12.3f\n", "Threshold + Hysteresis", t_threshold);
    printf("------------------------------------------\n");
    printf("%-28s %12.3f\n", "Total", total);

    free_rgb(&rgb);
    free_gray(&gray);
    free_gray(&blurred);
    free_gray(&magnitude);
    free_gray(&direction);
    free_gray(&suppressed);
    free_gray(&edges);

    return EXIT_SUCCESS;
}
