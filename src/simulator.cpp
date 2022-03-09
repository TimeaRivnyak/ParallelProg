#include "simulator.hpp"
#include "fmt/core.h"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <chrono>
#include <map>
#include <array>
#include <string>
#include <list>

std::map<std::string,std::array<double,3>> result_table;

void Simulator::setPrinting(bool toPrint) { printing = toPrint; }

void Simulator::initU() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid - 1); i++) {
        u[(i) * (grid + 1) + grid] = 1.0;
        u[(i) * (grid + 1) + grid - 1] = 1.0;
        #pragma omp parallel for
        for (SizeType j = 0; j < (grid - 1); j++) {
            u[(i) * (grid + 1) + j] = 0.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*(grid - 1)*3*4;
    if (result_table.count("initU") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["initU"] = data;
    } else {
        result_table["initU"][0]++;
        result_table["initU"][1] += time;
        result_table["initU"][2] += memory;
    }
}

void Simulator::initV() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid - 1); j++) {
            v[(i)*grid + j] = 0.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid)*(grid - 1)*4;
    if (result_table.count("initV") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["initV"] = data;
    } else {
        result_table["initV"][0]++;
        result_table["initV"][1] += time;
        result_table["initV"][2] += memory;
    }
}

void Simulator::initP() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid); j++) {
            p[(i) * (grid + 1) + j] = 1.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid)*(grid)*4;
    if (result_table.count("initP") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["initP"] = data;
    } else {
        result_table["initP"][0]++;
        result_table["initP"][1] += time;
        result_table["initP"][2] += memory;
    }
}

void Simulator::solveUMomentum(const FloatType Re) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (SizeType i = 1; i <= (grid - 2); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            un[(i) * (grid + 1) + j] = u[(i) * (grid + 1) + j]
                - dt
                    * ((u[(i + 1) * (grid + 1) + j] * u[(i + 1) * (grid + 1) + j] - u[(i - 1) * (grid + 1) + j] * u[(i - 1) * (grid + 1) + j]) / 2.0 / dx
                    + 0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
                            - (u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) * (v[(i + 1) * grid + j - 1] + v[(i)*grid + j - 1])) / dy)
                    - dt / dx * (p[(i + 1) * (grid + 1) + j] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                    * ((u[(i + 1) * (grid + 1) + j] - 2.0 * u[(i) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j]) / dx / dx
                     + (u[(i) * (grid + 1) + j + 1] - 2.0 * u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) / dy / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 2)*(grid - 1)*4*4;
    if (result_table.count("solveUMomentum") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["solveUMomentum"] = data;
    } else {
        result_table["solveUMomentum"][0]++;
        result_table["solveUMomentum"][1] += time;
        result_table["solveUMomentum"][2] += memory;
    }
}

void Simulator::applyBoundaryU() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType j = 1; j <= (grid - 1); j++) {
        un[(0) * (grid + 1) + j] = 0.0;
        un[(grid - 1) * (grid + 1) + j] = 0.0;
    }
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid - 1); i++) {
        un[(i) * (grid + 1) + 0] = -un[(i) * (grid + 1) + 1];
        un[(i) * (grid + 1) + grid] = 2 - un[(i) * (grid + 1) + grid - 1];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*2*4 + (grid - 1)*3*4;
    if (result_table.count("applyBoundaryU") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["applyBoundaryU"] = data;
    } else {
        result_table["applyBoundaryU"][0]++;
        result_table["applyBoundaryU"][1] += time;
        result_table["applyBoundaryU"][2] += memory;
    }
}

void Simulator::solveVMomentum(const FloatType Re) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 2); j++) {
            vn[(i)*grid + j] = v[(i)*grid + j]
                - dt * (0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
                              - (u[(i - 1) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i - 1) * grid + j])) / dx
                              + (v[(i)*grid + j + 1] * v[(i)*grid + j + 1] - v[(i)*grid + j - 1] * v[(i)*grid + j - 1]) / 2.0 / dy)
                              - dt / dy * (p[(i) * (grid + 1) + j + 1] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                              * ((v[(i + 1) * grid + j] - 2.0 * v[(i)*grid + j] + v[(i - 1) * grid + j]) / dx / dx
                              + (v[(i)*grid + j + 1] - 2.0 * v[(i)*grid + j] + v[(i)*grid + j - 1]) / dy / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*(grid - 2)*4*4;
    if (result_table.count("solveVMomentum") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["solveVMomentum"] = data;
    } else {
        result_table["solveVMomentum"][0]++;
        result_table["solveVMomentum"][1] += time;
        result_table["solveVMomentum"][2] += memory;
    }
}

void Simulator::applyBoundaryV() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType j = 1; j <= (grid - 2); j++) {
        vn[(0) * grid + j] = -vn[(1) * grid + j];
        vn[(grid)*grid + j] = -vn[(grid - 1) * grid + j];
    }
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid); i++) {
        vn[(i)*grid + 0] = 0.0;
        vn[(i)*grid + grid - 1] = 0.0;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 2)*3*4 + (grid)*2*4;
    if (result_table.count("applyBoundaryV") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["applyBoundaryV"] = data;
    } else {
        result_table["applyBoundaryV"][0]++;
        result_table["applyBoundaryV"][1] += time;
        result_table["applyBoundaryV"][2] += memory;
    }
}

void Simulator::solveContinuityEquationP(const FloatType delta) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            pn[(i) * (grid + 1) + j] = p[(i) * (grid + 1) + j]
                - dt * delta * ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*(grid - 1)*4*4;
    if (result_table.count("solveContinuityEquationP") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["solveContinuityEquationP"] = data;
    } else {
        result_table["solveContinuityEquationP"][0]++;
        result_table["solveContinuityEquationP"][1] += time;
        result_table["solveContinuityEquationP"][2] += memory;
    }
}

void Simulator::applyBoundaryP() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 1); i++) {
        pn[(i) * (grid + 1) + 0] = pn[(i) * (grid + 1) + 1];
        pn[(i) * (grid + 1) + grid] = pn[(i) * (grid + 1) + grid - 1];
    }
    #pragma omp parallel for
    for (SizeType j = 0; j <= (grid); j++) {
        pn[(0) * (grid + 1) + j] = pn[(1) * (grid + 1) + j];
        pn[(grid) * (grid + 1) + j] = pn[(grid - 1) * (grid + 1) + j];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*3*4 + (grid)*3*4;
    if (result_table.count("applyBoundaryP") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["applyBoundaryP"] = data;
    } else {
        result_table["applyBoundaryP"][0]++;
        result_table["applyBoundaryP"][1] += time;
        result_table["applyBoundaryP"][2] += memory;
    }
}

Simulator::FloatType Simulator::calculateError() {
    FloatType error = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2) reduction(+:error)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            m[(i) * (grid + 1) + j] =
                ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
            error += fabs(m[(i) * (grid + 1) + j]);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time = static_cast<double> (std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    double memory = (grid - 1)*(grid - 1)*4*4;
    if (result_table.count("calculateError") == 0){
        std::array<double,3> data = {1, time, memory};
        result_table["calculateError"] = data;
    } else {
        result_table["calculateError"][0]++;
        result_table["calculateError"][1] += time;
        result_table["calculateError"][2] += memory;
    }
    return error;
}

void Simulator::iterateU() {
    std::swap(u, un);
    // for (SizeType i = 0; i <= (grid - 1); i++) {
    //     for (SizeType j = 0; j <= (grid); j++) {
    //         u[(i) * (grid + 1) + j] = un[(i) * (grid + 1) + j];
    //     }
    // }
}

void Simulator::iterateV() {
    std::swap(v, vn);
    // for (SizeType i = 0; i <= (grid); i++) {
    //     for (SizeType j = 0; j <= (grid - 1); j++) {
    //         v[(i)*grid + j] = vn[(i)*grid + j];
    //     }
    // }
}

void Simulator::iterateP() {
    std::swap(p, pn);
    // for (SizeType i = 0; i <= (grid); i++) {
    //     for (SizeType j = 0; j <= (grid); j++) {
    //         p[(i) * (grid + 1) + j] = pn[(i) * (grid + 1) + j];
    //     }
    // }
}

void Simulator::deallocate() {
    // it doesn't do anything until we use vectors
    // because that deallocates automatically
    // but if we have to use a more raw data structure later it is needed
    // and when the the Tests overwrites some member those might won't deallocate
}

Simulator::Simulator(SizeType gridP)
    : grid([](auto g) {
          if (g <= 1) {
              throw std::runtime_error("Grid is smaller or equal to 1.0, give larger number");
          }
          return g;
      }(gridP)),
      dx(1.0 / static_cast<FloatType>(grid - 1)),
      dy(1.0 / static_cast<FloatType>(grid - 1)),
      dt(0.001 / std::pow(grid / 128.0 * 2.0, 2.0)),
      u(grid * (grid + 1)),
      un(grid * (grid + 1)),
      v((grid + 1) * grid),
      vn((grid + 1) * grid),
      p((grid + 1) * (grid + 1)),
      pn((grid + 1) * (grid + 1)),
      m((grid + 1) * (grid + 1)) {
    initU();
    initV();
    initP();
}

void Simulator::run(const FloatType delta, const FloatType Re, unsigned maxSteps) {
    if (printing) {
        fmt::print("Running simulation with delta: {}, Re: {}\n", delta, Re);
    }
    auto error = std::numeric_limits<FloatType>::max();
    unsigned step = 1;
    while (error > 0.00000001 && step <= maxSteps) {
        solveUMomentum(Re);
        applyBoundaryU();

        solveVMomentum(Re);
        applyBoundaryV();

        solveContinuityEquationP(delta);
        applyBoundaryP();

        error = calculateError();

        if (printing && (step % 1000 == 1)) {
            fmt::print("Error is {} for the step {}\n", error, step);
        }

        iterateU();
        iterateV();
        iterateP();
        ++step;
    }
    std::cout << "Name" << '\t' << "Count" << '\t' << "Time(s)" << '\t' << "Byte" << '\t' << "Byte(GB/s)" << std::endl;
    for (auto const& res : result_table)
    {
        std::cout << res.first << '\t' << res.second[0] << '\t' << res.second[1]*std::pow(10,(-9)) << '\t' << res.second[2] << '\t' << ((res.second[2]*8)/(res.second[1]*std::pow(10,(-9))))*std::pow(10,-10)*1.25 << std::endl;
    }
}

Simulator::~Simulator() { deallocate(); }