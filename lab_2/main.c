#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

// Ім’я файла вхідних даних 
const char *input_file_name = "in.txt";
// Ім’я файла результату
const char *output_file_name = "out.txt";

/* Обчислення функції */
double function(double x) {
    return pow(4, 2 * x + 3);
}

/* Перевірка правила Рунге */
bool check_Runge(double I2, double I, double epsilon) {
    return (fabs(I2 - I) / 3.) < epsilon;
}

/* Інтегрування методом лівих прямокутників */
double integrate_left_rectangle(double start, double finish, double epsilon) {
    int num_iterations = 1;
    /* Початкова кількість ітерацій */
    double last_res = 0.;
    /* Результат інтугрування на попередньому кроці */
    double res = -1.;
    /* Поточний результат інтегрування */
    double h = 0.;
    /* Ширина прямокутників */
    while (!check_Runge(res, last_res, epsilon)) {
        num_iterations *= 2;
        last_res = res;
        res = 0.;
        h = (finish - start) / num_iterations;
        for (int i = 0; i < num_iterations; i++) {
            res += function(start + i * h) * h;
        }
    }
    return res;
}

/* Запис одного числа типу double в файл з ім’ям filename */
void write_double_to_file(const char* filename, double data) {
    /* Відкриття файлу на запис */
    FILE *fp = fopen(filename, "w");
    /* Перевірка правильності відкриття файлу */
    if (fp == NULL) {
        printf("Failed to open the file");
        /* Завершення усіх процесів, що включені в комунікатор MPI_COMM_WORLD */
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Запис 1 числа типу double в файл */
    fprintf(fp, " % lg\n", data);
    /* Закриття файлу */
    fclose(fp);
}

int main(int argc, char* argv[]) {
    int np;
    int rank;

    /* Ініціалізація MPI */
    MPI_Init(&argc, &argv);

    /* Отримати кількість процесів */
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    /* Отримати індивідуальний ідентифікатор процесу */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 
     * Введення даних з файлу в масив з 3ох змінних. Відбувається у 
     * процесі з ідентифікатором 0. 
     * 1 - нижня межа інтегрування 
     * 2 - верхня межа інтегрування 
     * 3 - допустима похибка 
     */
    double input[3];

    if (rank == 0) {
        /* Відкриття файлу input.txt у режимі лише для читання */
        FILE *fp = fopen(input_file_name, "r");
        /* Перевірка правильності відкриття файлу */
        if (fp == NULL) {
            printf("Failed to open the file");
            /* Завершення усіх процесів, що включені в комунікатор MPI_COMM_WORLD */
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Зчитування 3 чисел типу double */
        for (int i = 0; i < 3; i++) fscanf(fp, " %lg", &input[i]);

        /* Закриття файлу */
        fclose(fp);
    }

    /* 
     * Передача введених даних від процесу 0 до всіх інших процесів, що 
     * включені у комунікатор MPI_COMM_WORLD 
     */
    MPI_Bcast(input, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = input[0];
    double finish = input[1];
    double epsilon = input[2];
    double step = (finish - start) / np;
    double res = integrate_left_rectangle(start + rank * step, start + (rank + 1) * step, epsilon / np);
    double result = res;

    /* 
     * Передача проміжного результату інтегрування від усіх процесів до процесу 0 
     * та збереження суми проміжних результатів у змінній result 
     */
    MPI_Reduce(&res, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /*
     * Виведення процесом 0 результату роботи програми у вихідний файл з ім’ям 
     * output.txt 
     */
    if (rank == 0) {
        write_double_to_file(output_file_name, result);
    }

    /* Закінчити роботу з MPI */
    MPI_Finalize();
    return 0;
}