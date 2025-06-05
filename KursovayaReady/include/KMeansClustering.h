//
//  KMeansClustering.h
//  KursovayaReady
//
//  Created by Никита Башлыков on 05.06.2025.
//

#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

using namespace std;

using Point = vector<double>;

class KMeansClustering {
private:
    vector<Point> data;         // Все точки для кластеризации
    size_t n_points;            // Число точек
    size_t dim;                 // Размерность пространства
    size_t k;                   // Число кластеров
    vector<Point> centroids;    // Центроиды
    vector<size_t> labels;      // Метки (кластер для каждой точки)

    // Евклидово расстояние
    double distance(const Point& a, const Point& b) const {
        double sum = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

public:
    // Конструктор
    KMeansClustering(const vector<Point>& input_data, size_t num_clusters)
        : data(input_data), n_points(input_data.size()), k(num_clusters) {
        if (n_points == 0) {
            throw invalid_argument("Набор данных пуст");
        }
        dim = data[0].size();
        for (const auto& pt : data) {
            if (pt.size() != dim) {
                throw invalid_argument("Все точки должны иметь одинаковую размерность");
            }
        }
        if (k == 0 || k > n_points) {
            throw invalid_argument("Неверное число кластеров k");
        }
        centroids.resize(k, Point(dim, 0.0));
        labels.resize(n_points, 0);
    }

    // Случайная инициализация центроидов
    void initialize_centroids_random() {
        mt19937 gen(random_device{}());
        uniform_int_distribution<size_t> dist(0, n_points - 1);
        unordered_set<size_t> chosen;
        size_t count = 0;
        while (count < k) {
            size_t idx = dist(gen);
            if (chosen.insert(idx).second) {
                centroids[count] = data[idx];
                ++count;
            }
        }
    }

    // Инициализация k-means++
    void initialize_centroids_kmeanspp() {
        mt19937 gen(random_device{}());
        uniform_int_distribution<size_t> dist0(0, n_points - 1);
        size_t first_idx = dist0(gen);
        centroids[0] = data[first_idx];

        vector<double> min_dists(n_points, numeric_limits<double>::max());

        for (size_t i = 1; i < k; ++i) {
            // Обновление min_dists
            for (size_t j = 0; j < n_points; ++j) {
                double dist_sq = 0.0;
                for (size_t dim_i = 0; dim_i < dim; ++dim_i) {
                    double diff = data[j][dim_i] - centroids[i - 1][dim_i];
                    dist_sq += diff * diff;
                }
                if (dist_sq < min_dists[j]) {
                    min_dists[j] = dist_sq;
                }
            }
            double sum_dists = 0.0;
            for (double dval : min_dists) {
                sum_dists += dval;
            }
            uniform_real_distribution<double> dist_real(0.0, sum_dists);
            double r = dist_real(gen);
            double cumulative = 0.0;
            size_t next_index = 0;
            for (size_t j = 0; j < n_points; ++j) {
                cumulative += min_dists[j];
                if (cumulative >= r) {
                    next_index = j;
                    break;
                }
            }
            centroids[i] = data[next_index];
        }
    }

    // Присвоение меток (каждая точка к ближайшему центроиду)
    void assign_labels() {
        for (size_t i = 0; i < n_points; ++i) {
            double best_dist = numeric_limits<double>::max();
            size_t best_cluster = 0;
            for (size_t c = 0; c < k; ++c) {
                double dist = distance(data[i], centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            labels[i] = best_cluster;
        }
    }

    // Пересчёт центроидов
    void update_centroids() {
        vector<Point> sum_coords(k, Point(dim, 0.0));
        vector<size_t> count_points(k, 0);

        for (size_t i = 0; i < n_points; ++i) {
            size_t cid = labels[i];
            for (size_t j = 0; j < dim; ++j) {
                sum_coords[cid][j] += data[i][j];
            }
            ++count_points[cid];
        }

        for (size_t c = 0; c < k; ++c) {
            if (count_points[c] == 0) {
                // Кластер пуст, оставляем предыдущий центроид
                continue;
            }
            for (size_t j = 0; j < dim; ++j) {
                centroids[c][j] = sum_coords[c][j] / static_cast<double>(count_points[c]);
            }
        }
    }

    // Вычисление суммы квадратов расстояний до центроидов (inertia)
    double compute_inertia() const {
        double inertia = 0.0;
        for (size_t i = 0; i < n_points; ++i) {
            size_t c = labels[i];
            double dist_sq = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                double diff = data[i][j] - centroids[c][j];
                dist_sq += diff * diff;
            }
            inertia += dist_sq;
        }
        return inertia;
    }

    // Основной метод: итерации до сходимости или достижения max_iter
    void run(size_t max_iter = 100, double tol = 1e-4, bool use_kpp = true) {
        if (use_kpp) {
            initialize_centroids_kmeanspp();
        } else {
            initialize_centroids_random();
        }

        double prev_inertia = numeric_limits<double>::max();
        for (size_t iter = 0; iter < max_iter; ++iter) {
            assign_labels();
            update_centroids();
            double curr_inertia = compute_inertia();
            if (fabs(prev_inertia - curr_inertia) < tol) {
                break;  // Сходимость достигнута
            }
            prev_inertia = curr_inertia;
        }
    }

    // Геттеры итоговых данных
    const vector<size_t>& get_labels() const {
        return labels;
    }

    const vector<Point>& get_centroids() const {
        return centroids;
    }
};
