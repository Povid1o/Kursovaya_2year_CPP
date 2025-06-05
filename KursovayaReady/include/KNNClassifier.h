//
//  KNNClassifier.h
//  KursovayaReady
//
//  Created by Никита Башлыков on 05.06.2025.
//

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

using namespace std;

using Point = vector<double>;

struct LabeledPoint {
    Point coordinates;  // Вектор признаков
    int label;          // Метка класса
};

class KNNClassifier {
private:
    vector<LabeledPoint> train_data;
    size_t dim;  // размерность признакового пространства

    // Евклидово расстояние
    double distance(const Point& a, const Point& b) const {
        if (a.size() != b.size()) {
            throw invalid_argument("Размерности векторов различаются");
        }
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

public:
    // Конструктор
    KNNClassifier(const vector<LabeledPoint>& data)
        : train_data(data) {
        if (train_data.empty()) {
            throw invalid_argument("Обучающая выборка пуста");
        }
        dim = train_data[0].coordinates.size();
        for (const auto& pt : train_data) {
            if (pt.coordinates.size() != dim) {
                throw invalid_argument("Все точки в обучающей выборке должны иметь одинаковую размерность");
            }
        }
    }

    // Предсказание метки для одной точки
    int predict(const Point& x, size_t k) const {
        if (x.size() != dim) {
            throw invalid_argument("Размерность тестовой точки не соответствует обучающей");
        }
        if (k == 0 || k > train_data.size()) {
            throw invalid_argument("Неверное значение k");
        }

        // Вектор расстояний
        vector<pair<double, size_t>> distances;
        distances.reserve(train_data.size());
        for (size_t i = 0; i < train_data.size(); ++i) {
            double dist = distance(x, train_data[i].coordinates);
            distances.emplace_back(dist, i);
        }

        // Нахождение k ближайших
        nth_element(distances.begin(),
                    distances.begin() + k,
                    distances.end(),
                    [](const auto& a, const auto& b) {
                        return a.first < b.first;
                    });

        // Подсчёт голосов
        unordered_map<int, size_t> vote_count;
        vote_count.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            int lbl = train_data[distances[i].second].label;
            vote_count[lbl]++;
        }

        // Определение метки с максимальным количеством голосов
        int best_label = train_data[distances[0].second].label;
        size_t max_votes = 0;
        for (const auto& kv : vote_count) {
            if (kv.second > max_votes) {
                max_votes = kv.second;
                best_label = kv.first;
            }
        }
        return best_label;
    }

    // Пакетное предсказание
    vector<int> batch_predict(const vector<Point>& test_data, size_t k) const {
        vector<int> predictions;
        predictions.reserve(test_data.size());
        for (const auto& x : test_data) {
            predictions.push_back(predict(x, k));
        }
        return predictions;
    }
};
