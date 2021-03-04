#include"pch.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

#define type double
#define e 2.7182818284590452353602874713526624977572
#define pi 3,14159265358979323
#define lam 1
#define a double(0)
#define b double(9.424777960769379)
type residual;

double x(int i, int size) {
	return a + i * (b - a) / size;
}
double K(double x, double s) {
	return cos(x+s);
}
double f(double x) {
	return (1-1.5*pi)*cos(x);
}
double u(double x) {
	return cos(x);
}

std::vector<std::vector<type>> make_A(int size)
// return zero vector
{
	std::vector<std::vector<type>> v(size);
	for (int i = 0; i < size; ++i) {
		v[i].resize(size);
		for (int j = 0; j < size; ++j) {
			if (i != j) {
				v[i][j] = -lam * (K(x(i, size), x(j + 1, size)) + K(x(i + 1, size), x(j + 1, size))) / size / 2;
			}
			else {
				v[i][j] = 1 - lam * (K(x(i, size), x(j + 1, size)) + K(x(i + 1, size), x(j + 1, size))) / 2 / size;
			}
		}
	}
	return v;
}

std::vector<type> make_uv(int size)
// return random fill vector
{
	std::vector<type> v(size);
	for (int i = 0; i < size; ++i)
		v[i] = (u(x(i, size)) + u(x(i + 1, size))) / 2;
	return v;
}
std::vector<type> make_fv(int size)
// return random fill vector
{
	std::vector<type> v(size);
	for (int i = 0; i < size; ++i)
		v[i] = (f(x(i, size)) + f(x(i + 1, size))) / 2;
	return v;
}
std::vector<type> matmul(std::vector<std::vector<type>>& A, std::vector<type>& y, int size)
// matrix by vector multiplication
{
	std::vector<type> result(size);
	type temp = 0;
#pragma omp parallel for private(temp)
	for (int i = 0; i < size; ++i)
	{
		temp = 0;
		for (int j = 0; j < size; ++j)
			temp += y[j] * A[i][j];
		result[i] = temp;
	}
	return result;
}
bool check(std::vector<type>& q, std::vector<type>& w, int size, type eps, double& error)
// checking result
{
	for (int i = 0; i < size; ++i) {
		if (error < abs(q[i] - w[i])) {
			error = abs(q[i] - w[i]);
		}
		if (abs(q[i] - w[i]) > eps)
			return false;
	}
	return true;
}
void recompute_accuracy(std::vector<std::vector<type>>& A, std::vector<type>& phi, std::vector<type>& f, int size, int number_threads)
// find accuracy using parallel
// accuracy metric is "The Frobenius norm"
{
	type temp;
	residual = 0;
	// omp_set_num_threads(number_threads);
 // works more effective than standart reduction
#pragma omp parallel for private(temp) shared(residual) schedule(dynamic)
	for (int i = 0; i < size; ++i)
	{
		temp = -f[i];
		for (int j = 0; j < size; ++j)
			temp += A[i][j] * phi[j];

#pragma  omp critical
		residual += temp * temp;
	}
	residual = sqrt(residual);
}

std::vector<type> low_sor(std::vector<std::vector<type>>& A, std::vector<type>& f, int size,
	type omega, type convergence_criteria, int number_threads, double& time)
	// function to find solution for Ay=f using lower SOR method
	// A - (size x size) matrix
	// vector f - right side of the equation
	// omega - parameter
	// residual - accuracy
{
	std::vector<type> y(size, 0);
	type sigma;
	double start = omp_get_wtime();         // set start time
	recompute_accuracy(A, y, f, size, number_threads); // find accuracy
	while (residual > convergence_criteria)
	{
		for (int i = 0; i < size; ++i)
		{
			sigma = -(A[i][i] * y[i]);
			for (int j = 0; j < size; ++j)
			{
				sigma += A[i][j] * y[j];
			}

			y[i] = (1 - omega) * y[i] + (omega / A[i][i]) * (f[i] - sigma);
		}
		recompute_accuracy(A, y, f, size, number_threads); // find accuracy
	}
	double end = omp_get_wtime(); // set end time
	time = end - start;              // compute time

	return y; // return result (x)
}

void make_report() {// ------------ Подготовительный блок -------------

	// ------------ Подготовительный блок -------------
	// Set parametres
	type omega = 0.9;
	type convergence_criteria = 0.01;
	// Generate matrix
	// file pointer
	std::ofstream fout("reportcard.csv");

	// opens an existing csv file or creates a new file.
	// Table Header
	fout << "Shape, 1 thread,2 threads,4 threads\n";
	double sum_time = 0;
	double time = 0; // time check variable
	int n = 20;      // number of repeats
	// ------------- Вычислительный блок -------------
	// Make Table
	for (int i = 128; i <= 2048; i *= 2)
	{//
		fout << i;
		std::vector<std::vector<type>> A = make_A(i);
		// Generate matrix x
		std::vector<type> uv = make_uv(i);
		std::vector<type> fv = make_fv(i);
		double time = 0, error, maxerror = 0;
		for (int j = 1; j < 5; j *= 2)
		{
			error = 0;
			sum_time = 0;
			for (int r = 0; r < n; ++r)
			{
				std::vector<type> y = low_sor(A, fv, i,                // matrix A and vector f
					omega, convergence_criteria,        // computetion parameters
					j, time);                                    // checkin parameters
				if (!check(y, uv, i, convergence_criteria, error)) // check results
				{
					std::cout << "Different results!!!\n";
					for (int k = 0; k < i; ++k)
						std::cout << y[k] << " " << uv[k] << std::endl;
					return;
				}
				sum_time += time;
				if (error > maxerror) maxerror = error;
				y.clear();
			}
			time = sum_time / n;
			fout << ", " << time;                            // Put parralel time
		}

		fout << ',' << maxerror << "\n";// end table string
		maxerror = 0;
		std::cout << i << std::endl; // Выводим счетчик для того чтобы наблюдать за процессом тестирования
		fv.clear();
		for (int k = 0; k < i; ++k) A[k].clear();
		A.clear(); uv.clear();
	}

}

int main(int argc, const char* argv[]) {
	make_report();
	return 0;
}
