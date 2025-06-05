//
//  main.cpp
//  KursovayaReady
//
//  Created by Никита Башлыков on 05.06.2025.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#include "KNNClassifier.h"
#include "KMeansClustering.h"

using namespace std;

using Point = vector<double>;
using LabeledPoint = ::LabeledPoint;

// Чтение обучающей выборки для k-NN
vector<LabeledPoint> read_labeled_data(const string& filename, size_t& dim_out) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<LabeledPoint> data;
    string line;
    size_t dim = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        vector<double> coords;
        while (getline(ss, token, ',')) {
            coords.push_back(stod(token));
        }
        if (coords.empty()) {
            throw runtime_error("Пустая строка в файле: " + filename);
        }
        int label = static_cast<int>(coords.back());
        coords.pop_back();
        if (dim == 0) {
            dim = coords.size();
        } else if (coords.size() != dim) {
            throw runtime_error("Несоответствие размерностей в файле: " + filename);
        }
        data.push_back({coords, label});
    }
    if (data.empty()) {
        throw runtime_error("Файл с обучающими данными пуст: " + filename);
    }
    dim_out = dim;
    return data;
}

// Чтение тестовых данных (без меток) для k-NN
vector<Point> read_unlabeled_data(const string& filename, size_t dim) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<Point> data;
    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        Point coords;
        while (getline(ss, token, ',')) {
            coords.push_back(stod(token));
        }
        if (coords.size() != dim) {
            throw runtime_error("Несоответствие размерностей в файле: " + filename);
        }
        data.push_back(coords);
    }
    if (data.empty()) {
        throw runtime_error("Файл с тестовыми данными пуст: " + filename);
    }
    return data;
}

// Чтение точек для k-means
vector<Point> read_points_data(const string& filename, size_t& dim_out) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<Point> data;
    string line;
    size_t dim = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        Point coords;
        while (getline(ss, token, ',')) {
            coords.push_back(stod(token));
        }
        if (coords.empty()) {
            throw runtime_error("Пустая строка в файле: " + filename);
        }
        if (dim == 0) {
            dim = coords.size();
        } else if (coords.size() != dim) {
            throw runtime_error("Несоответствие размерностей в файле: " + filename);
        }
        data.push_back(coords);
    }
    if (data.empty()) {
        throw runtime_error("Файл с данными для кластеризации пуст: " + filename);
    }
    dim_out = dim;
    return data;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            cerr << "Недостаточно аргументов.\n";
            cerr << "Использование:\n";
            cerr << "  Для k-NN:\n"
                 << "    " << argv[0] << " knn train.csv test.csv k\n";
            cerr << "  Для k-means:\n"
                 << "    " << argv[0] << " kmeans data.csv k [max_iter] [tol]\n";
            return EXIT_FAILURE;
        }
        string mode = argv[1];
        if (mode == "knn") {
            if (argc != 5) {
                throw invalid_argument("Ожидается 4 аргумента для k-NN: knn train.csv test.csv k");
            }
            string train_file = argv[2];
            string test_file = argv[3];
            size_t k = stoul(argv[4]);

            size_t dim_train = 0;
            auto train_data = read_labeled_data(train_file, dim_train);
            auto test_data = read_unlabeled_data(test_file, dim_train);

            KNNClassifier knn(train_data);
            auto predictions = knn.batch_predict(test_data, k);

            cout << "Результаты классификации (k = " << k << "):\n";
            for (size_t i = 0; i < test_data.size(); ++i) {
                const auto& pt = test_data[i];
                cout << "Точка [";
                for (size_t j = 0; j < pt.size(); ++j) {
                    cout << pt[j];
                    if (j + 1 < pt.size()) cout << ", ";
                }
                cout << "] => Метка: " << predictions[i] << "\n";
            }
        }
        else if (mode == "kmeans") {
            if (argc < 4 || argc > 6) {
                throw invalid_argument("Ожидается: kmeans data.csv k [max_iter] [tol]");
            }
            string data_file = argv[2];
            size_t k = stoul(argv[3]);
            size_t max_iter = 100;
            double tol = 1e-4;
            if (argc >= 5) {
                max_iter = stoul(argv[4]);
            }
            if (argc == 6) {
                tol = stod(argv[5]);
            }

            size_t dim_data = 0;
            auto data = read_points_data(data_file, dim_data);

            KMeansClustering kmeans(data, k);
            kmeans.run(max_iter, tol, true);

            const auto& centroids = kmeans.get_centroids();
            const auto& labels = kmeans.get_labels();

            cout << "Результаты кластеризации (k = " << k << "):\n";
            cout << "Центроиды:\n";
            for (size_t c = 0; c < centroids.size(); ++c) {
                cout << "Центроид " << c << ": [";
                for (size_t j = 0; j < centroids[c].size(); ++j) {
                    cout << centroids[c][j];
                    if (j + 1 < centroids[c].size()) cout << ", ";
                }
                cout << "]\n";
            }

            cout << "Принадлежность точек к кластерам:\n";
            for (size_t i = 0; i < data.size(); ++i) {
                cout << "Точка [";
                for (size_t j = 0; j < data[i].size(); ++j) {
                    cout << data[i][j];
                    if (j + 1 < data[i].size()) cout << ", ";
                }
                cout << "] => Кластер: " << labels[i] << "\n";
            }
        }
        else {
            cerr << "Неподдерживаемый режим: " << mode << "\n";
            return EXIT_FAILURE;
        }
    }
    catch (const exception& ex) {
        cerr << "Ошибка: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
