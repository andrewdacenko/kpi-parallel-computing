#include <mpi.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <stdarg.h> 
#include <stdbool.h>
#include <math.h>

struct my_matrix {
    int rows;
    int cols;
    double *data;
};

struct my_vector {
    int size;
    double *data;
};

void write_vector(const char *filename, struct my_vector *vec);

void fatal_error(const char *message, int errorcode) {
    printf("fatal error : code %d, %s\n", errorcode, message);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, errorcode);
}

struct my_matrix *matrix_alloc(int rows, int cols, double initial) {
    struct my_matrix *result = (struct my_matrix*) malloc(sizeof (struct my_matrix));

    result->rows = rows;
    result->cols = cols;
    result->data = (double*) malloc(sizeof (double) * rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[i * cols + j] = initial;
        }
    }

    return result;
}

struct my_vector *vector_alloc(int size, double initial) {
    struct my_vector *result = (struct my_vector*) malloc(sizeof (struct my_vector));

    result->size = size;
    result->data = (double*) malloc(sizeof (double) * size);

    for (int i = 0; i < size; i++) {
        result->data[i] = initial;
    }

    return result;
}

void matrix_print(const char *filename, struct my_matrix *mat) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fatal_error("cant write to file", 2);
    }

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            fprintf(f, "%lf ", mat->data[i * mat->cols + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void write_vector(const char *filename, struct my_vector *vec) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fatal_error("cant write to file", 2);
    }

    for (int i = 0; i < vec->size; i++) {
        fprintf(f, "%lf ", vec->data[i]);
    }
    fprintf(f, "\n");

    fclose(f);
}

struct my_matrix *read_matrix(const char *filename) {
    FILE *mat_file = fopen(filename, "r");
    if (mat_file == NULL) {
        fatal_error("can�t open matrix file", 1);
    }

    int rows;
    int cols;

    fscanf(mat_file, "%d %d", &rows, &cols);

    struct my_matrix *result = matrix_alloc(rows, cols, 0.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(mat_file, "%lf", &result->data[i * cols + j]);
        }
    }

    fclose(mat_file);
    return result;
}

struct my_vector *read_vector(const char *filename) {
    FILE *vec_file = fopen(filename, "r");
    if (vec_file == NULL) {
        fatal_error("can�t open vector file", 1);
    }

    int size;

    fscanf(vec_file, "%d", &size);

    struct my_vector *result = vector_alloc(size, 0.0);

    for (int i = 0; i < size; i++) {
        fscanf(vec_file, "%lf", &result->data[i]);
    }

    fclose(vec_file);
    return result;
}


/* Точність обчислення коренів */
const double epsilon = 0.001;

/* Функція обчислення наступного наближення ітераційного процесу Якобі */
void jacobi_iteration(
        int start_row, // Номер першого рядка частини матриці 
        struct my_matrix *mat_A_part, // Частина рядків матриці коефіціентів 
        struct my_vector *b, // Вектор вільних членів 
        struct my_vector *vec_prev_x, // Вектор попереднього наближення 
        struct my_vector *vec_next_x_part, // Частина вектору наступного наближення (встановлюється в функції) 
        double *residue_norm_part) // Значення норми на попередньому кроці обчислень (встановлюється в функції) 
{
    /* Акумулятор значення норми даної частини обчислень */
    double my_residue_norm_part = 0.0;

    /* Поелементне обчислення частини вектору наступного наближення */
    for (int i = 0; i < vec_next_x_part->size; i++) {
        double sum = 0.0;
        for (int j = 0; j < mat_A_part->cols; j++) {
            if (i + start_row != j) {
                sum += mat_A_part->data[i * mat_A_part->cols + j] * vec_prev_x->data[j];
            }
        }
        sum = b->data[i + start_row] - sum;
        vec_next_x_part->data[i] = sum / mat_A_part->data[i * mat_A_part->cols + i + start_row];

        /* Обчислення норми на попередньому кроці */
        sum = -sum + mat_A_part->data[i * mat_A_part->cols + i + start_row] * vec_prev_x->data[i + start_row];
        my_residue_norm_part += sum * sum;
    }

    *residue_norm_part = my_residue_norm_part;
}

/* Основна функція */
int main(int argc, char *argv[]) {
    const char *input_file_MA = "MA.txt";
    const char *input_file_b = "b.txt";
    const char *output_file_x = "out.txt";

    /* Ініціалізація MPI */
    MPI_Init(&argc, &argv);

    /* Отримання загальної кількості задач та рангу поточної задачі */
    int np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Зчитування даних в задачі 0 */
    struct my_matrix *MA;
    struct my_vector *b;
    int N;

    if (rank == 0) {
        MA = read_matrix(input_file_MA);
        b = read_vector(input_file_b);

        if (MA->rows != MA->cols) {
            fatal_error("Matrix is not square!", 4);
        }
        if (MA->rows != b->size) {
            fatal_error("Dimensions of matrix and vector don’t match!", 5);
        }

        N = b->size;
    }

    /* Розсилка всім задачам розмірності матриць та векторів */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Виділення пам’яті для зберігання вектора вільних членів */
    if (rank != 0) {
        b = vector_alloc(N, .0);
    }

    /* 
     * Обчислення частини векторів та матриці, яка буде зберігатися в кожній 
     * задачі, вважаемо що N = k*np. Виділення пам’яті для зберігання частин
     * векторів та матриць в кожній задачі та встановлення їх початкових значень 
     */
    int part = N / np;
    struct my_matrix *MAh = matrix_alloc(part, N, .0);
    struct my_vector *oldx = vector_alloc(N, .0);
    struct my_vector *newx = vector_alloc(part, .0);

    /* 
     * Розбиття вихідної матриці MA на частини по part рядків та розсилка частин 
     * у всі задачі. Звільнення пам’яті, виділеної для матриці МА. 
     */
    if (rank == 0) {
        MPI_Scatter(MA->data, N*part, MPI_DOUBLE, MAh->data, N*part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(MA);
    } else {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, MAh->data, N*part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Розсилка вектора вільних членів */
    MPI_Bcast(b->data, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 
     * Обчислення норми вектору вільних членів в задачі 0 та розсилка її значення 
     * у всі задачі 
     */
    double b_norm = 0.0;
    if (rank == 0) {
        for (int i = 0; i < b->size; i++) {
            b_norm = b->data[i] * b->data[i];
        }
        b_norm = sqrt(b_norm);
    }
    MPI_Bcast(&b_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Значення критерію зупинки ітерації */
    double last_stop_criteria;

    /* Основний цикл ітерації Якобі */
    for (int i = 0; i < 1000; i++) {
        double residue_norm_part;
        double residue_norm;

        jacobi_iteration(rank * part, MAh, b, oldx, newx, &residue_norm_part);

        /* Обчислення сумарного значення нев’язки */
        MPI_Allreduce(&residue_norm_part, &residue_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residue_norm = sqrt(residue_norm);

        /* 
         * Перевірка критерію зупинки ітерації. Оскільки на поточному кроці 
         * обчислюється значення норми для попереднього кроку, то результатом 
         * обчислення є дані попереднього кроку 
         */
        last_stop_criteria = residue_norm / b_norm;
        if (last_stop_criteria < epsilon) {
            break;
        }

        /* Збір значень поточного наближення вектору невідомих */
        MPI_Allgather(newx->data, part, MPI_DOUBLE, oldx->data, part, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /* Вивід результату */
    if (rank == 0) {
        write_vector(output_file_x, oldx);
    }

    /* Повернення виділених ресурсів системі та фіналізація середовища MPI */
    free(MAh);
    free(oldx);
    free(newx);
    free(b);

    return MPI_Finalize();
}