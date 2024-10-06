#include "dataset.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <LittleFS.h>  // Include LittleFS library

// Assuming Neuron is defined somewhere in dataset.h
void Dataset::load_dataset() {
    // Open catalog using LittleFS
    File catalog_file = LittleFS.open(catalog.c_str(), "r");
    if (!catalog_file) {
        Serial.println("Failed to open catalog file");
        return;
    }

    std::string line, word;
    std::vector<std::string> headers;

    // Variables for tracking the longest sequence
    int max_length = 0;

    // Reading headers from the catalog file
    if (catalog_file.available()) {
        line = catalog_file.readStringUntil('\n').c_str();  // Read the header
        std::stringstream ss(line);
        while (std::getline(ss, word, ',')) {
            headers.push_back(word);
        }
    }

    double min_time_rel = std::numeric_limits<double>::max();
    double max_time_rel = std::numeric_limits<double>::lowest();
    double min_velocity = std::numeric_limits<double>::max();
    double max_velocity = std::numeric_limits<double>::lowest();

    // First pass: find min, max values and the longest sequence length
    while (catalog_file.available()) {
        line = catalog_file.readStringUntil('\n').c_str();
        std::stringstream ss(line);
        std::vector<std::string> row;

        while (std::getline(ss, word, ',')) {
            row.push_back(word);
        }

        std::string filename = row[0];
        if (LittleFS.exists(("/data/training/" + filename + ".csv").c_str())) {
            File training_file = LittleFS.open(("/data/training/" + filename + ".csv").c_str(), "r");

            if (training_file) {
                training_file.readStringUntil('\n');  // Skip header

                int sequence_length = 0;
                while (training_file.available()) {
                    training_file.readStringUntil('\n');
                    sequence_length++;
                }

                max_length = std::max(max_length, sequence_length / 7);  // Update max length
                training_file.close();
            }
        }
    }

    // Reset catalog file pointer for the second pass
    catalog_file.seek(0, SeekSet);
    catalog_file.readStringUntil('\n');  // Skip the header

    // Second pass: normalize and pad the sequences
    while (catalog_file.available()) {
        line = catalog_file.readStringUntil('\n').c_str();
        std::stringstream ss(line);
        std::vector<std::string> row;

        while (std::getline(ss, word, ',')) {
            row.push_back(word);
        }

        std::string filename = row[0];
        std::string mq_type = row[4]; 
        double time_rel = std::stod(row[2]);

        if (LittleFS.exists(("/data/training/" + filename + ".csv").c_str())) {
            detection_time_rel.push_back(time_rel);

            File training_file = LittleFS.open(("/data/training/" + filename + ".csv").c_str(), "r");
            std::vector<std::vector<Neuron>> current_data;

            if (training_file) {
                training_file.readStringUntil('\n');  // Skip the header

                int count = 0;
                double time_average = 0.0;
                double velocity_average = 0.0;

                while (training_file.available()) {
                    line = training_file.readStringUntil('\n').c_str();
                    std::stringstream data_stream(line);
                    std::vector<std::string> data;
                    std::string data_point;

                    while (std::getline(data_stream, data_point, ',')) {
                        data.push_back(data_point);
                    }

                    time_average += std::stod(data[1]);
                    velocity_average += std::stod(data[2]);

                    if (count % 7 == 6) {
                        Neuron time_neuron, velocity_neuron;
                        time_neuron.input = time_average / 7;
                        velocity_neuron.input = velocity_average / 7;

                        std::vector<Neuron> current_neurons = {time_neuron, velocity_neuron};
                        current_data.push_back(current_neurons);

                        time_average = 0.0;
                        velocity_average = 0.0;
                    }

                    count++;
                }

                while (current_data.size() < max_length) {
                    Neuron zero_time_neuron, zero_velocity_neuron;
                    zero_time_neuron.input = 0.0;
                    zero_velocity_neuron.input = 0.0;
                    current_data.push_back({zero_time_neuron, zero_velocity_neuron});
                }

                data.push_back(current_data);
                training_file.close();
            }
        }
    }

    catalog_file.close();  // Close catalog file after processing
}

void Dataset::shuffle_dataset() {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<std::vector<Neuron>>> shuffled_data;
    std::vector<double> shuffled_detection_time_rel;

    for (int i : indices) {
        shuffled_data.push_back(data[i]);
        shuffled_detection_time_rel.push_back(detection_time_rel[i]);
    }

    data = shuffled_data;
    detection_time_rel = shuffled_detection_time_rel;
}
